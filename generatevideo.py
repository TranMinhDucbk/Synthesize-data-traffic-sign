import cv2
import numpy as np
import os
import random
import glob
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================

INPUT_VIDEO_DIR = 'background'       # Folder chứa video gốc
ICON_DIR_ROOT = 'sign'               # Folder chứa biển báo
OUTPUT_VIDEO_DIR = 'output_videos'   # Nơi lưu video kết quả (.mp4)

# --- Cấu hình hiệu ứng ---
SPAWN_CHANCE = 0.1         # 10% cơ hội xuất hiện biển mới
MAX_ACTIVE_SIGNS = 3       # Tối đa 3 biển cùng lúc
SIGN_LIFE_DURATION = 60    # Biển tồn tại 60 frames

# --- Priority ---
PRIORITY_CLASSES = ['bien_can_doc_theo_cot', 'bien_can_doc_theo_hang', 'bien_can_doc_theo_hang_khong_mui_ten']
PRIORITY_RATIO = 0.15      

# --- Kích thước & Biến đổi ---
START_SCALE = 0.02
END_SCALE = 0.25
OBJECT_AUG_CHANCE = 0.7
MAX_ROTATION_ANGLE = 5
CLEAN_EDGES = True
ERODE_ITERS = 1

# ==============================================================================
# 2. CÁC HÀM HỖ TRỢ (GIỮ NGUYÊN)
# ==============================================================================

def auto_convert_to_bgra(image):
    if image is None: return None
    if len(image.shape) == 2:
        bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        gray = image.copy()
    elif len(image.shape) == 3:
        if image.shape[2] == 4: return image
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: return image
    _, mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    bgra[:, :, 3] = cv2.bitwise_not(mask)
    return bgra

def clean_icon_edges(icon_img, erode_iters=1):
    if icon_img.shape[2] < 4: return icon_img
    b, g, r, a = cv2.split(icon_img)
    _, alpha_thresh = cv2.threshold(a, 200, 255, cv2.THRESH_BINARY)
    if erode_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        alpha_clean = cv2.erode(alpha_thresh, kernel, iterations=erode_iters)
    else: alpha_clean = alpha_thresh
    return cv2.merge((b, g, r, alpha_clean))

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_TRANSPARENT)

def augment_object(obj_img):
    if random.random() > OBJECT_AUG_CHANCE: return obj_img
    if obj_img.shape[2] < 4: return obj_img
    obj_bgr = obj_img[:, :, :3]
    obj_alpha = obj_img[:, :, 3]
    augmented_bgr = obj_bgr.copy()
    
    # Sáng/Tối
    if random.random() > 0.5:
        brightness = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(augmented_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, int((brightness - 1) * 255))
        v = np.clip(v, 0, 255)
        augmented_bgr = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    
    # Làm mờ
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        augmented_bgr = cv2.GaussianBlur(augmented_bgr, (ksize, ksize), 0)
        
    augmented_obj_img = cv2.merge((augmented_bgr, obj_alpha))
    
    # Xoay
    if random.random() > 0.5:
        angle = random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
        augmented_obj_img = rotate_image(augmented_obj_img, angle)
        
    return augmented_obj_img

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = overlay.shape[:2]
    x1, y1 = x, y
    x2, y2 = x + obj_w, y + obj_h
    if x1 >= bg_w or y1 >= bg_h or x2 <= 0 or y2 <= 0: return background
    
    x1_c, y1_c = max(0, x1), max(0, y1)
    x2_c, y2_c = min(bg_w, x2), min(bg_h, y2)
    w_c, h_c = x2_c - x1_c, y2_c - y1_c
    if w_c <= 0 or h_c <= 0: return background
    
    ox1, oy1 = x1_c - x1, y1_c - y1
    overlay_crop = overlay[oy1:oy1+h_c, ox1:ox1+w_c]
    bg_crop = background[y1_c:y2_c, x1_c:x2_c]
    
    overlay_bgr = overlay_crop[:, :, :3]
    mask = (overlay_crop[:, :, 3] / 255.0).astype(np.float32)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    bg_crop = (bg_crop.astype(np.float32) * (1.0 - mask_3d)) + (overlay_bgr.astype(np.float32) * mask_3d)
    background[y1_c:y2_c, x1_c:x2_c] = bg_crop.astype(np.uint8)
    return background

# ==============================================================================
# 3. CLASS & MAIN
# ==============================================================================

def load_dataset_structure(root_dir):
    all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    priority_pool = []
    normal_pool = []
    print(f"🔄 Đang load icons từ {root_dir}...")
    for cls_name in all_classes:
        cls_dir = os.path.join(root_dir, cls_name)
        images = glob.glob(os.path.join(cls_dir, '*.png')) + glob.glob(os.path.join(cls_dir, '*.jpg'))
        pool_list = priority_pool if cls_name in PRIORITY_CLASSES else normal_pool
        for img_path in images:
            pool_list.append(img_path)
    return priority_pool, normal_pool

class MovingSign:
    def __init__(self, img_path, video_w, video_h, duration):
        # Đọc ảnh
        raw_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        raw_img = auto_convert_to_bgra(raw_img)
        if CLEAN_EDGES: raw_img = clean_icon_edges(raw_img, ERODE_ITERS)
        self.original_img = augment_object(raw_img)
        
        self.total_frames = duration
        self.current_frame = 0
        self.is_dead = False
        
        self.start_x = video_w // 2 + random.randint(-50, 50)
        self.start_y = video_h // 2 - random.randint(20, 100)
        side = 1 if random.random() > 0.5 else -1
        self.end_x = video_w // 2 + (side * video_w * 0.45)
        self.end_y = video_h + 100 

    def update(self):
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.is_dead = True
            return None, None, None
        progress = self.current_frame / self.total_frames
        progress_curve = progress * progress 
        cur_x = int(self.start_x + (self.end_x - self.start_x) * progress_curve)
        cur_y = int(self.start_y + (self.end_y - self.start_y) * progress_curve)
        cur_scale = START_SCALE + (END_SCALE - START_SCALE) * progress_curve
        return cur_x, cur_y, cur_scale

def main():
    if not os.path.exists(OUTPUT_VIDEO_DIR):
        os.makedirs(OUTPUT_VIDEO_DIR)
        
    priority_pool, normal_pool = load_dataset_structure(ICON_DIR_ROOT)
    if not priority_pool and not normal_pool: print("❌ Không có icons!"); return
    
    video_files = glob.glob(os.path.join(INPUT_VIDEO_DIR, '*.mp4')) + glob.glob(os.path.join(INPUT_VIDEO_DIR, '*.avi'))
    
    print(f"🎥 Tìm thấy {len(video_files)} video nền.")

    for vid_path in video_files:
        filename = os.path.basename(vid_path)
        output_path = os.path.join(OUTPUT_VIDEO_DIR, f"result_{filename}")
        
        cap = cv2.VideoCapture(vid_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- CÀI ĐẶT VIDEO WRITER ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        active_signs = []
        
        print(f"🎬 Đang render: {filename} -> {output_path}")
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Sinh biển mới
                if len(active_signs) < MAX_ACTIVE_SIGNS and random.random() < SPAWN_CHANCE:
                    dice = random.random()
                    pool = priority_pool if (dice < PRIORITY_RATIO and priority_pool) else (normal_pool or priority_pool)
                    rand_icon = random.choice(pool)
                    new_sign = MovingSign(rand_icon, w, h, SIGN_LIFE_DURATION)
                    if new_sign.original_img is not None:
                        active_signs.append(new_sign)
                
                # Vẽ biển
                for sign in active_signs[::-1]:
                    x, y, scale = sign.update()
                    if sign.is_dead:
                        active_signs.remove(sign); continue
                    
                    h_icon, w_icon = sign.original_img.shape[:2]
                    new_h, new_w = int(h * scale), int(h * scale * (w_icon / h_icon))
                    
                    if new_h > 0 and new_w > 0:
                        icon_resized = cv2.resize(sign.original_img, (new_w, new_h))
                        frame = overlay_transparent(frame, icon_resized, x - new_w//2, y - new_h//2)
                
                # --- QUAN TRỌNG: LƯU VÀO VIDEO ---
                out.write(frame)
                pbar.update(1)
                
        cap.release()
        out.release() # Nhớ đóng file
        
    print(f"\n🎉 XONG HẾT RỒI! Vào folder '{OUTPUT_VIDEO_DIR}' xem phim nhé!")

if __name__ == "__main__":
    main()