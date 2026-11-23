import os
os.environ["YOLO_VERBOSE"] = "False" 

import time
import queue
import threading
import warnings
import logging
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

# 必須寫在 import ultralytics 之後，才不會被覆蓋掉
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ==========================================
# 全域設定區（以後改照片/影片、調精準度與防抖參數都在這裡！）
# ==========================================
# 1. 影像來源：可填單張圖片路徑、影片路徑 (.mp4/.avi/.mov)，或填 0 開啟攝影機
SOURCE = "d:/畢業專題/test/test_media/IMG_3571.MOV"
# SOURCE = "d:/畢業專題/test/IMG_3230.JPG"
# SOURCE = 0

# 2. 各物件判斷精準度 (信心度門檻：0.0 ~ 1.0，越高越嚴格、越低越容易抓到)
CONF_OBSTACLE   = 0.45   # 一般障礙物（行人、汽機車等）
CONF_SIDEWALK   = 0.15   # 人行道
CONF_CROSSWALK  = 0.22   # 斑馬線
CONF_LIGHT      = 0.08   # 行人號誌
CONF_YOLO_WORLD = 0.25   # 特殊障礙物（施工三角錐、水坑、導盲磚、樓梯）

# 3. 系統效能、排程與語音設定
DEBUG_MODE = False         # True 才會存檔 debug_light.jpg 與印出號誌除錯
LIGHT_INTERVAL = 3         # 行人號誌每幾幀跑一次（第 0, 3, 6, 9... 幀）
LA_INTERVAL = 6            # YOLO-World 每幾幀跑一次（錯開在第 1, 7, 13... 幀，與號誌零撞車）
SPEECH_COOLDOWN = 3.5      # 即時模式下，語音播報最短間隔 (秒)
USE_YOLO_WORLD = True      # 保留三角錐等特殊障礙物偵測

# 4. 畫面與決策穩定度設定（防閃爍、平滑化、近距離過濾）
NO_PATH_BUFFER = 3         # 連續幾幀沒偵測到步道才切換成 NO_PATH_FOUND
LIGHT_CONFIRM_FRAMES = 2   # 紅/綠燈需連續確認幾幀才生效
ANGLE_SMOOTH_ALPHA = 0.35  # 路徑角度平滑係數 (0.1~1.0，越小越平滑、越大反應越快)
NEAR_OBSTACLE_RATIO = 0.55 # 障礙物底部超過畫面高度此比例，才觸發近距離語音警告

# ==========================================
# 非同步語音播報執行緒（避免語音卡住畫面）
# ==========================================
class AsyncSpeaker:
    def __init__(self, cooldown=3.5):
        self.cooldown = cooldown
        self.last_spoken_time = 0.0
        self.last_text = ""
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while self.running:
            try:
                text = self.q.get(timeout=0.5)
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
            except queue.Empty:
                continue

    def speak(self, text, force=False):
        now = time.time()
        if not force:
            if (now - self.last_spoken_time) < self.cooldown and text == self.last_text:
                return
            if (now - self.last_spoken_time) < 1.8:
                return
        self.last_spoken_time = now
        self.last_text = text
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.q.put(text)

    def stop(self):
        self.running = False
        self.q.put(None)

# ==========================================
# 時序狀態追蹤器（負責防閃爍緩衝與角度平滑）
# ==========================================
class NavStateTracker:
    def __init__(self):
        self.no_path_count = 0
        self.last_valid_surface = "none"
        self.last_valid_pts = []
        self.smoothed_angle = None

        self.pending_light = None
        self.light_streak = 0
        self.confirmed_light = None
        self.no_light_count = 0

    def update_light(self, raw_light_status, is_single_image=False):
        if is_single_image:
            return raw_light_status

        if raw_light_status in ("RED", "GREEN"):
            self.no_light_count = 0
            if raw_light_status == self.pending_light:
                self.light_streak += 1
            else:
                self.pending_light = raw_light_status
                self.light_streak = 1

            if self.light_streak >= LIGHT_CONFIRM_FRAMES:
                self.confirmed_light = raw_light_status
        else:
            self.no_light_count += 1
            # 連續幾幀都沒看到號誌才清除狀態，避免短暫漏抓
            if self.no_light_count >= LIGHT_INTERVAL + 1:
                self.pending_light = None
                self.light_streak = 0
                self.confirmed_light = None

        return self.confirmed_light

    def smooth_angle(self, raw_angle, is_single_image=False):
        if raw_angle is None:
            return self.smoothed_angle if not is_single_image else None
        if is_single_image or self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            self.smoothed_angle = round(
                ANGLE_SMOOTH_ALPHA * raw_angle + (1.0 - ANGLE_SMOOTH_ALPHA) * self.smoothed_angle,
                1
            )
        return self.smoothed_angle

# ==========================================
# 輔助函式：計算重疊率 (IoU)
# ==========================================
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ==========================================
# 功能函式 1：OpenCV 行人號誌顏色判斷 (含邊界保護 + 過曝 LED 支援)
# ==========================================
def detect_light_color(img, box, debug=False):
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(img_w - 1, int(x1)))
    y1 = max(0, min(img_h - 1, int(y1)))
    x2 = max(0, min(img_w, int(x2)))
    y2 = max(0, min(img_h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN", "狀態不明"

    crop_img = img[y1:y2, x1:x2]
    if crop_img is None or crop_img.size == 0:
        return "UNKNOWN", "狀態不明"

    h, w = crop_img.shape[:2]
    if w >= 6 and h >= 6:
        pad_x, pad_y = int(w * 0.15), int(h * 0.08)
        inner = crop_img[pad_y:max(pad_y + 1, h - pad_y), pad_x:max(pad_x + 1, w - pad_x)]
    else:
        inner = crop_img

    if inner.size == 0 or inner.shape[0] < 2:
        inner = crop_img

    mid_y = max(1, inner.shape[0] // 2)
    top_half = inner[:mid_y, :].astype(np.int16)
    bottom_half = inner[mid_y:, :].astype(np.int16)

    if top_half.size == 0 or bottom_half.size == 0:
        return "PED-LIGHT", "行人號誌"

    b_bot, g_bot, r_bot = bottom_half[:, :, 0], bottom_half[:, :, 1], bottom_half[:, :, 2]
    green_color_mask = (g_bot > 110) & ((g_bot - r_bot) > 15)
    bright_led_mask = (g_bot > 180) & (b_bot > 150) & (g_bot >= r_bot - 20)
    green_pixels = int(np.sum(green_color_mask | bright_led_mask))

    b_top, g_top, r_top = top_half[:, :, 0], top_half[:, :, 1], top_half[:, :, 2]
    red_mask = ((r_top > 110) & ((r_top - g_top) > 20)) | ((r_top > 130) & ((r_top - b_top) > 35))
    red_pixels = int(np.sum(red_mask))

    if debug:
        print(f"🚦 號誌除錯 | 框尺寸:{w}x{h} | 上半紅像素:{red_pixels} | 下半亮/綠像素:{green_pixels}")
        cv2.imwrite(
            "d:/畢業專題/test/debug_light.jpg",
            cv2.resize(crop_img, (120, 200), interpolation=cv2.INTER_NEAREST)
        )

    if green_pixels >= 2:
        return "GREEN", "行人綠燈"
    elif red_pixels >= 2:
        return "RED", "行人紅燈"
    else:
        return "PED-LIGHT", "行人號誌"

# ==========================================
# 功能函式 2：相對位置精算
# ==========================================
def calculate_relative_position(box_xyxy, img_w, img_h, fov_deg=70.0):
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    norm_x = (cx - (img_w / 2.0)) / (img_w / 2.0)
    angle_deg = norm_x * (fov_deg / 2.0)

    if angle_deg < -20: clock_str = "10點鐘"
    elif angle_deg < -7: clock_str = "11點鐘"
    elif angle_deg <= 7: clock_str = "12點鐘"
    elif angle_deg <= 20: clock_str = "1點鐘"
    else: clock_str = "2點鐘"

    urgency = max(0.1, min(1.0, y2 / float(img_h)))
    if angle_deg < -7: lra_left, lra_right = int(urgency * 100), 0
    elif angle_deg > 7: lra_left, lra_right = 0, int(urgency * 100)
    else: lra_left, lra_right = int(urgency * 100), int(urgency * 100)

    return round(angle_deg, 1), clock_str, lra_left, lra_right

# ==========================================
# 功能函式 3：本地幾何「最佳安全路徑線」計算 (含遮罩去雜訊)
# ==========================================
def compute_safe_path(img_h, img_w, detections, target_surface="sidewalk"):
    walk_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    has_surface = False

    for det in detections:
        if det['group'] == target_surface and det.get('polygon') is not None:
            cv2.fillPoly(walk_mask, det['polygon'], 255)
            has_surface = True

    if not has_surface:
        return [], None, False

    margin = 15
    for det in detections:
        if det['group'] in ['obstacle', 'la']:
            x1, y1, x2, y2 = det['box']
            ground_y1 = int(y1 + (y2 - y1) * 0.5)
            cv2.rectangle(
                walk_mask,
                (max(0, x1 - margin), max(0, ground_y1)),
                (min(img_w, x2 + margin), min(img_h, y2 + margin)),
                0,
                -1
            )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    walk_mask = cv2.morphologyEx(walk_mask, cv2.MORPH_OPEN, kernel)
    walk_mask = cv2.morphologyEx(walk_mask, cv2.MORPH_CLOSE, kernel)

    ys = np.where(np.any(walk_mask > 0, axis=1))[0]
    if len(ys) < 15:
        return [], None, True

    y_bottom = int(ys[-1]) - 5
    y_top = int(ys[0]) + 10
    if y_bottom <= y_top:
        return [], None, True

    sample_ys = np.linspace(y_bottom, y_top, 12, dtype=int)
    raw_points = []

    for y in sample_ys:
        row = (walk_mask[y, :] > 0).astype(np.int8)
        diff = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) == 0:
            continue

        widths = ends - starts
        best_idx = int(np.argmax(widths))
        if widths[best_idx] >= 15:
            mid_x = int((starts[best_idx] + ends[best_idx]) / 2)
            raw_points.append((mid_x, int(y)))

    if len(raw_points) < 2:
        return raw_points, None, True

    smooth_points = []
    for i in range(len(raw_points)):
        xs = [raw_points[j][0] for j in range(max(0, i - 1), min(len(raw_points), i + 2))]
        smooth_points.append((int(np.mean(xs)), raw_points[i][1]))

    lookahead_pt = smooth_points[min(len(smooth_points) - 1, len(smooth_points) // 2)]
    norm_x = (lookahead_pt[0] - (img_w / 2.0)) / (img_w / 2.0)
    path_angle = round(norm_x * 35.0, 1)

    return smooth_points, path_angle, False

# ==========================================
# 功能函式 4：純本地安全路徑決策引擎（含防閃爍、角度平滑、近距離過濾）
# ==========================================
def local_navigation_decision(img_h, img_w, detections, state, is_single_image=False):
    has_sidewalk = any(d['group'] == 'sidewalk' and d.get('polygon') is not None for d in detections)
    has_crosswalk = any(d['group'] == 'crosswalk' and d.get('polygon') is not None for d in detections)

    has_light = False
    raw_light_status = None
    for d in detections:
        if d['group'] == 'light':
            has_light = True
            if d.get('light_color_en') == 'RED':
                raw_light_status = 'RED'
                break
            elif d.get('light_color_en') == 'GREEN':
                raw_light_status = 'GREEN'

    # 號誌防閃爍確認
    light_status = state.update_light(raw_light_status, is_single_image=is_single_image)

    # 步道防閃爍緩衝
    using_buffered_path = False
    if has_sidewalk or has_crosswalk:
        state.no_path_count = 0
    else:
        state.no_path_count += 1
        if not is_single_image and state.no_path_count < NO_PATH_BUFFER and state.last_valid_surface != "none":
            using_buffered_path = True

    if light_status == 'RED' and has_crosswalk:
        action = "WAIT_RED_LIGHT"
        target_surface = "sidewalk" if has_sidewalk else "crosswalk"
    elif light_status == 'GREEN' and has_crosswalk:
        action = "CROSS_STREET"
        target_surface = "crosswalk"
    elif has_sidewalk:
        action = "WALK_SIDEWALK"
        target_surface = "sidewalk"
    elif has_crosswalk:
        action = "CROSS_CAUTION"
        target_surface = "crosswalk"
    elif using_buffered_path:
        target_surface = state.last_valid_surface
        action = "WALK_SIDEWALK" if target_surface == "sidewalk" else "CROSS_CAUTION"
    else:
        action = "NO_PATH_FOUND"
        target_surface = "none"

    safe_pts, raw_angle, is_blocked = [], None, False
    if using_buffered_path:
        safe_pts = state.last_valid_pts
        raw_angle = None
    elif target_surface in ["sidewalk", "crosswalk"]:
        safe_pts, raw_angle, is_blocked = compute_safe_path(img_h, img_w, detections, target_surface)
        if len(safe_pts) >= 2:
            state.last_valid_surface = target_surface
            state.last_valid_pts = safe_pts

    if is_blocked and action != "WAIT_RED_LIGHT":
        action = "STOP_OBSTACLE"

    # 路徑角度 EMA 平滑化
    path_angle = state.smooth_angle(raw_angle, is_single_image=is_single_image)

    if path_angle is None:
        steer_en, steer_tw = "HOLD", "保持原地"
    elif path_angle < -20:
        steer_en, steer_tw = "10 O'CLOCK", "往10點鐘方向前進"
    elif path_angle < -7:
        steer_en, steer_tw = "11 O'CLOCK", "往11點鐘方向前進"
    elif path_angle <= 7:
        steer_en, steer_tw = "12 O'CLOCK", "往12點鐘方向直走"
    elif path_angle <= 20:
        steer_en, steer_tw = "1 O'CLOCK", "往1點鐘方向前進"
    else:
        steer_en, steer_tw = "2 O'CLOCK", "往2點鐘方向前進"

    # 近距離障礙物過濾：只抓前方 (11~1點鐘) 且底部超過畫面 NEAR_OBSTACLE_RATIO 的近距威脅，並優先報最近的
    near_y_threshold = int(img_h * NEAR_OBSTACLE_RATIO)
    front_obstacles = [
        d for d in detections
        if d['group'] in ['obstacle', 'la']
        and d['clock_dir'] in ["11點鐘", "12點鐘", "1點鐘"]
        and d['box'][3] >= near_y_threshold
    ]
    front_obstacles.sort(key=lambda d: d['box'][3], reverse=True)
    obs_warn = f"，注意前方有{front_obstacles[0]['tw_name']}" if front_obstacles else ""

    if action == "WAIT_RED_LIGHT":
        speech = f"前方行人紅燈，請留在人行道稍候{obs_warn}"
    elif action == "CROSS_STREET":
        speech = f"前方行人綠燈，請沿斑馬線{steer_tw}{obs_warn}"
    elif action == "STOP_OBSTACLE":
        speech = f"前方路徑受阻，請先停下確認{obs_warn}"
    elif action == "WALK_SIDEWALK":
        speech = f"請沿人行道{steer_tw}{obs_warn}"
    elif action == "CROSS_CAUTION":
        if has_light:
            speech = f"前方有斑馬線與行人號誌，但燈色不明，請確認來車後{steer_tw}"
        else:
            speech = f"前方有斑馬線，未偵測到號誌，請確認來車後{steer_tw}"
    else:
        speech = f"未偵測到安全步道，請減速慢行{obs_warn}"

    return {
        "action": action,
        "surface": target_surface,
        "path_pts": safe_pts,
        "path_angle": path_angle,
        "steer_en": steer_en,
        "speech": speech
    }

# ==========================================
# 模型預載函式：OpenVINO 加速版
# ==========================================
def load_all_models():
    print("🚀 預先載入 OpenVINO 加速模型...")
    models = {
        "bg": YOLO("yolo26s_openvino_model", task="detect"),
        "surface": None,
        "light": None,
        "la": None,
        "locate_targets": {
            "traffic cone": "施工三角錐",
            "puddle": "水坑",
            "tactile paving": "導盲磚",
            "stairs": "樓梯"
        }
    }

    surface_path = "d:/畢業專題/test/surface_seg_v26_openvino_model"
    if os.path.exists(surface_path):
        models["surface"] = YOLO(surface_path, task="segment")
    else:
        print(f"⚠️ 找不到合併路面 OpenVINO 模型：{surface_path}")

    light_path = "d:/畢業專題/test/light_model_v26_openvino_model"
    if os.path.exists(light_path):
        models["light"] = YOLO(light_path, task="detect")
    else:
        print(f"⚠️ 找不到行人號誌 OpenVINO 模型：{light_path}")

    if USE_YOLO_WORLD:
        model_la = YOLO("yolov8s-worldv2.pt")
        model_la.set_classes(list(models["locate_targets"].keys()))
        models["la"] = model_la

    print("✅ 所有模型載入完成！")
    return models

# ==========================================
# 單幀推論與繪圖主流程（錯開排程 + 分頻快取）
# ==========================================
def process_frame(img, models, state, frame_idx=0, cache=None, is_single_image=False):
    if cache is None:
        cache = {"light": [], "la": []}

    img_h, img_w = img.shape[:2]
    combined_img = img.copy()
    overlay = img.copy()
    raw_detections = []

    tw_dict = {
        "person": "行人", "car": "汽車", "motorcycle": "機車", "bus": "公車",
        "bicycle": "腳踏車", "truck": "大貨車", "stop sign": "停車標誌"
    }

    # --- 1. 官方背景模型（OpenVINO，每幀都跑）---
    model_bg = models["bg"]
    res_bg = model_bg(img, conf=CONF_OBSTACLE, imgsz=480, verbose=False)
    for box in res_bg[0].boxes:
        cls_id = int(box.cls[0])
        en_name = model_bg.names[cls_id]
        if en_name not in tw_dict:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        angle, clock_dir, lra_l, lra_r = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
        raw_detections.append({
            'box': (x1, y1, x2, y2),
            'conf': float(box.conf[0]),
            'group': 'obstacle',
            'en_name': en_name,
            'tw_name': tw_dict[en_name],
            'is_custom': False,
            'color': (255, 160, 0),
            'clock_dir': clock_dir,
            'angle': angle,
            'lra': (lra_l, lra_r),
            'polygon': None,
            'requires_fill': False
        })

    # --- 2. 合併路面分割模型（OpenVINO，每幀都跑：0=人行道, 1=斑馬線）---
    model_surface = models["surface"]
    if model_surface is not None:
        min_surf_conf = min(CONF_SIDEWALK, CONF_CROSSWALK)
        res_surf = model_surface(img, conf=min_surf_conf, imgsz=480, verbose=False)
        has_masks = res_surf[0].masks is not None
        for i, box in enumerate(res_surf[0].boxes):
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])

            if cls_id == 0:
                if conf_val < CONF_SIDEWALK:
                    continue
                group_cat, label_en, label_tw, color = 'sidewalk', 'SIDEWALK', '人行道', (0, 255, 0)
            elif cls_id == 1:
                if conf_val < CONF_CROSSWALK:
                    continue
                group_cat, label_en, label_tw, color = 'crosswalk', 'CROSSWALK', '斑馬線', (255, 0, 255)
            else:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            angle, clock_dir, _, _ = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
            polygon_pts = np.int32([res_surf[0].masks.xy[i]]) if has_masks else None

            raw_detections.append({
                'box': (x1, y1, x2, y2),
                'conf': conf_val,
                'group': group_cat,
                'en_name': label_en,
                'tw_name': label_tw,
                'is_custom': True,
                'color': color,
                'clock_dir': clock_dir,
                'angle': angle,
                'lra': (0, 0),
                'polygon': polygon_pts,
                'requires_fill': True
            })

    # --- 3. 行人號誌模型（OpenVINO，排程：第 0, 3, 6, 9... 幀）---
    if is_single_image or (frame_idx % LIGHT_INTERVAL == 0):
        light_dets = []
        if models["light"] is not None:
            roi_img = img[0:int(img_h * 0.55), :]
            res_light = models["light"](roi_img, conf=CONF_LIGHT, imgsz=736, verbose=False)
            for box in res_light[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) >= (y2 - y1):
                    continue
                angle, clock_dir, _, _ = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
                color_en, color_tw = detect_light_color(img, (x1, y1, x2, y2), debug=DEBUG_MODE)
                light_dets.append({
                    'box': (x1, y1, x2, y2),
                    'conf': float(box.conf[0]),
                    'group': 'light',
                    'en_name': 'PED_LIGHT',
                    'tw_name': '行人號誌',
                    'light_color_en': color_en,
                    'light_color_tw': color_tw,
                    'is_custom': True,
                    'color': (0, 255, 255),
                    'clock_dir': clock_dir,
                    'angle': angle,
                    'lra': (0, 0),
                    'polygon': None,
                    'requires_fill': False
                })
        cache["light"] = light_dets

    # --- 4. YOLO-World（320px，錯開排程：第 1, 7, 13, 19... 幀，與號誌零撞車）---
    if is_single_image or (frame_idx % LA_INTERVAL == 1):
        la_dets = []
        if models["la"] is not None:
            model_la = models["la"]
            locate_targets = models["locate_targets"]
            res_la = model_la(img, conf=CONF_YOLO_WORLD, imgsz=320, verbose=False)
            for box in res_la[0].boxes:
                cls_id = int(box.cls[0])
                en_word = model_la.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                angle, clock_dir, lra_l, lra_r = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
                la_dets.append({
                    'box': (x1, y1, x2, y2),
                    'conf': float(box.conf[0]),
                    'group': 'la',
                    'en_name': en_word,
                    'tw_name': locate_targets[en_word],
                    'is_custom': False,
                    'color': (255, 255, 0),
                    'clock_dir': clock_dir,
                    'angle': angle,
                    'lra': (lra_l + 20, lra_r + 20),
                    'polygon': None,
                    'requires_fill': False
                })
        cache["la"] = la_dets

    raw_detections.extend(cache["light"])
    raw_detections.extend(cache["la"])

    # --- 5. NMS 消除重疊框 ---
    raw_detections.sort(key=lambda x: (x['is_custom'], x['conf']), reverse=True)
    final_detections = []
    for new_det in raw_detections:
        keep = True
        for existing_det in final_detections:
            if new_det['group'] == existing_det['group']:
                if calculate_iou(new_det['box'], existing_det['box']) > 0.35:
                    keep = False
                    break
        if keep:
            final_detections.append(new_det)

    # --- 6. 純本地最佳安全路徑決策（含防閃爍與角度平滑）---
    nav = local_navigation_decision(img_h, img_w, final_detections, state, is_single_image=is_single_image)
    safe_path_pts = nav["path_pts"]

    # --- 7. 繪圖輸出 ---
    max_lra_left, max_lra_right = 0, 0
    for det in final_detections:
        if det.get('requires_fill') and det.get('polygon') is not None:
            cv2.fillPoly(overlay, det['polygon'], det['color'])

    combined_img = cv2.addWeighted(overlay, 0.45, combined_img, 0.55, 0)

    if len(safe_path_pts) >= 2:
        pts_array = np.array(safe_path_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(combined_img, [pts_array], isClosed=False, color=(0, 0, 0), thickness=10)
        cv2.polylines(combined_img, [pts_array], isClosed=False, color=(255, 255, 255), thickness=5)
        cv2.arrowedLine(
            combined_img, safe_path_pts[-2], safe_path_pts[-1],
            (255, 255, 255), thickness=6, tipLength=0.45
        )

    for det in final_detections:
        x1, y1, x2, y2 = det['box']
        angle = det['angle']
        lra_l, lra_r = det['lra']
        max_lra_left = max(max_lra_left, lra_l)
        max_lra_right = max(max_lra_right, lra_r)

        if det['group'] == 'light':
            tag_text = f"PED_{det['light_color_en']} | {angle:+}deg"
        else:
            tag_text = f"{det['en_name']} | {angle:+}deg"

        font_scale, thickness = 0.65, 2
        (text_w, text_h), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        if det.get('requires_fill') and det.get('polygon') is not None:
            cv2.polylines(combined_img, [det['polygon']], isClosed=True, color=det['color'], thickness=3)
            M = cv2.moments(det['polygon'])
            if M["m00"] != 0:
                text_x = int(M["m10"] / M["m00"]) - (text_w // 2)
                text_y = int(M["m01"] / M["m00"]) + (text_h // 2)
            else:
                text_x, text_y = x1, max(y1 - 10, 25)
        else:
            cv2.rectangle(combined_img, (x1, y1), (x2, y2), det['color'], 3 if det['is_custom'] else 2)
            text_x, text_y = x1, max(y1 - 8, 25)

        text_x = max(2, min(text_x, img_w - text_w - 2))
        text_y = max(text_h + 5, min(text_y, img_h - 5))
        cv2.rectangle(combined_img, (text_x - 3, text_y - text_h - 4), (text_x + text_w + 3, text_y + baseline), det['color'], -1)
        cv2.putText(combined_img, tag_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    nav["lra"] = (min(100, max_lra_left), min(100, max_lra_right))
    return combined_img, nav

# ==========================================
# 主程式進入點（自動判斷圖片或即時影片/攝影機）
# ==========================================
if __name__ == "__main__":
    models = load_all_models()
    speaker = AsyncSpeaker(cooldown=SPEECH_COOLDOWN)
    state = NavStateTracker()
    win_name = "YOLO Local Safe Path Navigation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 950, 680)

    is_image = isinstance(SOURCE, str) and SOURCE.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    if is_image:
        img = cv2.imread(SOURCE)
        if img is None:
            print(f"❌ 找不到照片：{SOURCE}")
            exit()

        print("🔥 正在進行 OpenVINO 首次暖機編譯...")
        process_frame(img, models, state, frame_idx=0, is_single_image=True)

        t0 = time.time()
        result_img, nav = process_frame(img, models, state, frame_idx=0, is_single_image=True)
        infer_ms = (time.time() - t0) * 1000
        fps = 1000.0 / max(infer_ms, 1.0)

        lra_l, lra_r = nav["lra"]
        angle_str = f"{nav['path_angle']:+}deg" if nav["path_angle"] is not None else "N/A"
        print(f"⚡ 真實單幀耗時：{infer_ms:.0f} ms (理論 {fps:.1f} FPS)")
        print(f"📳 LRA 震動：左 {lra_l}% | 右 {lra_r}%")
        print(f"🧭 本地決策：{nav['action']} | 目標路面：{nav['surface']} | 路徑角度：{angle_str}")
        print(f"🔊 語音播報：{nav['speech']}")

        cv2.imwrite("d:/畢業專題/test/auto_nav_local_path_result.jpg", result_img)
        cv2.imshow(win_name, result_img)
        speaker.speak(nav["speech"], force=True)
        cv2.waitKey(0)

    else:
        cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            print(f"❌ 無法開啟影像來源：{SOURCE}")
            exit()

        is_webcam = isinstance(SOURCE, int)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 120:
            video_fps = 30.0

        frame_idx = 0
        current_video_frame = 0
        start_time = None
        cache = {"light": [], "la": []}
        print("🎥 即時導航啟動！按鍵盤 'q' 鍵結束。")

        while True:
            # 如果是影片檔，根據真實時間自動跳幀，維持 1 倍正常速度
            if not is_webcam and start_time is not None:
                target_frame = int((time.time() - start_time) * video_fps)
                while current_video_frame < target_frame:
                    if not cap.grab():
                        break
                    current_video_frame += 1

            ret, frame = cap.read()
            if not ret:
                break
            current_video_frame += 1

            t0 = time.time()
            result_img, nav = process_frame(frame, models, state, frame_idx=frame_idx, cache=cache, is_single_image=False)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            frame_idx += 1

            # 第一幀跑完（暖機完成）才開始算影片同步時間，避免開頭被跳過
            if start_time is None:
                start_time = time.time()

            cv2.putText(result_img, f"FPS: {fps:.1f} | {nav['action']}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow(win_name, result_img)
            speaker.speak(nav["speech"])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    speaker.stop()
    cv2.destroyAllWindows()