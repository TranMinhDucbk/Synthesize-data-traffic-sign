import cv2
import numpy as np
import os
import random
import glob
import yaml
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================

# --- Đường dẫn Input ---
BG_DIR = 'background'
OBJ_DIR_ROOT = 'sign' 

# --- Đường dẫn Output ---
BASE_OUTPUT_DIR = 'dataset_final_clean_priority'

# --- 3 Class "VIP" (Ưu tiên xuất hiện 60%) ---
PRIORITY_CLASSES = ['bien_can_doc_theo_cot', 'bien_can_doc_theo_hang', 'bien_can_doc_1_o' , 'bien_can_doc_theo_hang_khong_mui_ten', 'bien_1_mui_ten_1_o']
PRIORITY_RATIO = 0.15

# --- Thông số sinh ảnh ---
NUM_IMAGES_TO_GENERATE = 10000     # Tổng số ảnh muốn tạo
TRAIN_VAL_SPLIT_RATIO = 0.8     # 80% Train, 20% Val
MIN_OBJECTS_PER_IMAGE = 2
MAX_OBJECTS_PER_IMAGE = 5
MIN_SCALE = 0.05                # 5% chiều cao ảnh nền
MAX_SCALE = 0.2                 # 20% chiều cao ảnh nền

# --- Augmentation ---
OBJECT_AUG_CHANCE = 0.7         
MAX_BLUR_KERNEL = 5            
MIN_BRIGHTNESS = 0.7
MAX_BRIGHTNESS = 1.3
MAX_ROTATION_ANGLE = 15         
MIN_CONTRAST = 0.5  # 0.5 là giảm tương phản (ảnh nhờ nhờ, sương mù)
MAX_CONTRAST = 1.5  # 1.5 là tăng tương phản (ảnh gắt)

# --- Rửa ảnh ---
CLEAN_EDGES = True              
ERODE_ITERS = 1                 

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ ẢNH
# ==============================================================================

def auto_convert_to_bgra(image):
    """
    [CẬP NHẬT] Hàm tự động xử lý mọi loại ảnh (Đen trắng, RGB, RGBA)
    tránh lỗi IndexError và tự động xóa nền trắng.
    """
    if image is None:
        return None

    # 1. Xử lý ảnh Đen Trắng (Grayscale - chỉ có 2 chiều)
    if len(image.shape) == 2:
        # Chuyển thẳng từ Xám sang BGRA (4 kênh)
        bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        gray = image.copy() # Dùng luôn ảnh gốc để làm mask
        
    # 2. Xử lý ảnh Có Màu (3 hoặc 4 chiều)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            # Đã là ảnh có kênh Alpha trong suốt -> Không cần xử lý thêm
            return image 
        elif image.shape[2] == 3:
            # Ảnh màu bình thường (JPG) -> Chuyển sang BGRA
            bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image # Trường hợp dị thường, trả về gốc
    else:
        return image # Trường hợp dị thường, trả về gốc

    # 3. Tạo mặt nạ xóa nền trắng
    # Những điểm ảnh >= 225 (màu gần trắng) sẽ thành màu trắng (255)
    _, mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    
    # 4. Những chỗ nền trắng (mask=255) thì gán Alpha = 0 (cho nó trong suốt)
    bgra[:, :, 3] = cv2.bitwise_not(mask)
    
    return bgra

def clean_icon_edges(icon_img, erode_iters=1):
    """Làm sạch viền icon"""
    if icon_img.shape[2] < 4:
        return icon_img

    b, g, r, a = cv2.split(icon_img)
    _, alpha_thresh = cv2.threshold(a, 200, 255, cv2.THRESH_BINARY)

    if erode_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        alpha_clean = cv2.erode(alpha_thresh, kernel, iterations=erode_iters)
    else:
        alpha_clean = alpha_thresh

    return cv2.merge((b, g, r, alpha_clean))

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_TRANSPARENT), M

def augment_object(obj_img):
    if random.random() > OBJECT_AUG_CHANCE:
        return obj_img 

    if obj_img.shape[2] < 4: return obj_img
    obj_bgr = obj_img[:, :, :3]
    obj_alpha = obj_img[:, :, 3]

    augmented_bgr = obj_bgr.copy()

    # Sáng/Tối
    if random.random() > 0.5:
        brightness = random.uniform(MIN_BRIGHTNESS, MAX_BRIGHTNESS)
        hsv = cv2.cvtColor(augmented_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, int((brightness - 1) * 255))
        v = np.clip(v, 0, 255)
        augmented_bgr = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    # 2. Tương phản (Contrast) - [MỚI THÊM]
    if random.random() > 0.5:
        contrast = random.uniform(MIN_CONTRAST, MAX_CONTRAST)
        # Hàm convertScaleAbs: (pixel * alpha) + beta
        # Ở đây ta chỉ đổi alpha (tương phản), giữ beta=0 (sáng/tối đã làm ở trên)
        augmented_bgr = cv2.convertScaleAbs(augmented_bgr, alpha=contrast, beta=0)

    # Làm Mờ
    if random.random() > 0.5:
        ksize = random.choice(range(3, MAX_BLUR_KERNEL + 1, 2)) 
        augmented_bgr = cv2.GaussianBlur(augmented_bgr, (ksize, ksize), 0)
    
    augmented_obj_img = cv2.merge((augmented_bgr, obj_alpha))

    # Xoay
    if random.random() > 0.5:
        angle = random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
        augmented_obj_img, _ = rotate_image(augmented_obj_img, angle)
    
    return augmented_obj_img

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = overlay.shape[:2]
    
    x1, y1 = x, y
    x2, y2 = x + obj_w, y + obj_h

    if x1 < 0 or y1 < 0 or x2 > bg_w or y2 > bg_h:
        return background, None 

    overlay_bgr = overlay[:, :, :3]
    
    # [LƯU Ý] Đoạn này sẽ không còn lỗi vì hàm auto_convert_to_bgra đã đảm bảo có kênh 3
    mask = (overlay[:, :, 3] / 255.0).astype(np.float32)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    roi = background[y1:y2, x1:x2]

    try:
        blended = (roi.astype(np.float32) * (1.0 - mask_3d)) + (overlay_bgr.astype(np.float32) * mask_3d)
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
        return background, (x1, y1, x2, y2)
    except Exception:
        return background, None

# ==============================================================================
# 3. LOGIC QUẢN LÝ DATASET
# ==============================================================================

def load_dataset_structure(root_dir):
    all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
    
    priority_pool = []
    normal_pool = []
    
    print(f"🔄 Đang quét dữ liệu từ {root_dir}...")
    
    for cls_name in all_classes:
        cls_dir = os.path.join(root_dir, cls_name)
        cls_id = class_to_id[cls_name]
        
        images = glob.glob(os.path.join(cls_dir, '*.png')) + glob.glob(os.path.join(cls_dir, '*.jpg'))
        
        if cls_name in PRIORITY_CLASSES:
            for img_path in images:
                priority_pool.append((img_path, cls_id))
        else:
            for img_path in images:
                normal_pool.append((img_path, cls_id))
                
    return class_to_id, priority_pool, normal_pool

# ==============================================================================
# 4. MAIN PROGRAM
# ==============================================================================

def main():
    # --- [MỚI] MENU LỰA CHỌN CHẾ ĐỘ ---
    print("="*50)
    print("🚦 CHỌN CHẾ ĐỘ SINH DỮ LIỆU:")
    print(f"   1. Sinh tập Train & Val (Số lượng: {NUM_IMAGES_TO_GENERATE} ảnh)")
    print(f"   2. Sinh tập Test (Số lượng: {int(NUM_IMAGES_TO_GENERATE * 0.1)} ảnh - 10%)")
    print("="*50)
    
    choice = input("👉 Nhập lựa chọn của bạn (1 hoặc 2): ").strip()
    
    if choice == '2':
        is_test_mode = True
        actual_num_images = max(1, int(NUM_IMAGES_TO_GENERATE * 0.1)) # Sinh 10% ảnh
        target_splits = ['test']
        print(f"\n🛠️ Đã chọn: Sinh tập TEST ({actual_num_images} ảnh).")
    else:
        is_test_mode = False
        actual_num_images = NUM_IMAGES_TO_GENERATE # Sinh 100% ảnh
        target_splits = ['train', 'val']
        print(f"\n🛠️ Đã chọn: Sinh tập TRAIN & VAL ({actual_num_images} ảnh).")

    # --- BƯỚC 1: Load dữ liệu ---
    if not os.path.exists(OBJ_DIR_ROOT):
        print(f"❌ Lỗi: Không tìm thấy thư mục icons tại {OBJ_DIR_ROOT}")
        return

    class_to_id, priority_pool, normal_pool = load_dataset_structure(OBJ_DIR_ROOT)
    
    print(f"✅ Đã tải xong: {len(class_to_id)} Class | VIP: {len(priority_pool)} | Normal: {len(normal_pool)}")

    backgrounds = glob.glob(os.path.join(BG_DIR, '*.jpg')) + glob.glob(os.path.join(BG_DIR, '*.png'))
    if not backgrounds:
        print(f"❌ Lỗi: Không tìm thấy ảnh nền trong {BG_DIR}")
        return

    # --- BƯỚC 2: Tạo thư mục Output theo chế độ đã chọn ---
    # Luôn tạo sẵn 3 thư mục để file data.yaml không bị lỗi khi tìm kiếm
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, 'labels', split), exist_ok=True)

    # --- BƯỚC 3: Sinh ảnh ---
    print(f"🚀 Bắt đầu sinh {actual_num_images} ảnh...")
    
    for i in tqdm(range(actual_num_images)):
        
        dice = random.random()
        if dice < PRIORITY_RATIO and len(priority_pool) > 0:
            current_pool = priority_pool
        else:
            current_pool = normal_pool if len(normal_pool) > 0 else priority_pool

        bg_path = random.choice(backgrounds)
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        bg_h, bg_w = bg_img.shape[:2]
        
        yolo_labels = []
        num_objects = random.randint(MIN_OBJECTS_PER_IMAGE, MAX_OBJECTS_PER_IMAGE)

        for _ in range(num_objects):
            if not current_pool: break 
            
            obj_path, class_id = random.choice(current_pool)
            
            obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj_img is None: continue

            obj_img = auto_convert_to_bgra(obj_img)

            if CLEAN_EDGES:
                obj_img = clean_icon_edges(obj_img, erode_iters=ERODE_ITERS)

            obj_img = augment_object(obj_img)
            
            scale = random.uniform(MIN_SCALE, MAX_SCALE)
            h, w = obj_img.shape[:2]
            new_h = int(bg_h * scale)
            if new_h == 0: continue
            new_w = int(new_h * (w / h))
            obj_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if bg_w <= new_w or bg_h <= new_h: continue
            x = random.randint(0, bg_w - new_w)
            y = random.randint(0, bg_h - new_h)
            
            bg_img, bbox = overlay_transparent(bg_img, obj_resized, x, y)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                xc = ((x1 + x2) / 2) / bg_w
                yc = ((y1 + y2) / 2) / bg_h
                wn = (x2 - x1) / bg_w
                hn = (y2 - y1) / bg_h
                yolo_labels.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if yolo_labels:
            # === [MỚI] KIỂM TRA CHẾ ĐỘ ĐỂ LƯU VÀO ĐÚNG THƯ MỤC ===
            if is_test_mode:
                split = 'test'
            else:
                # Nếu là Train/Val thì chia theo tỷ lệ (mặc định < 0.8 là train)
                split = 'train' if random.random() < TRAIN_VAL_SPLIT_RATIO else 'val'
            # ====================================================
            
            # Đặt tên file khác nhau giữa Train và Test để tránh bị ghi đè nếu chạy nhiều lần
            prefix = "test" if is_test_mode else "train"
            img_name = f"gen_{prefix}_{i}.jpg"
            txt_name = f"gen_{prefix}_{i}.txt"
            
            cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, 'images', split, img_name), bg_img)
            with open(os.path.join(BASE_OUTPUT_DIR, 'labels', split, txt_name), 'w') as f:
                f.write('\n'.join(yolo_labels))

    # --- BƯỚC 4: Tạo data.yaml (Luôn khai báo đủ 3 tập) ---
    print("\n📝 Đang cập nhật file data.yaml...")
    yaml_data = {
        'path': os.path.abspath(BASE_OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  # Bổ sung Test vào đây
        'nc': len(class_to_id),
        'names': {id: name for name, id in class_to_id.items()}
    }
    with open(os.path.join(BASE_OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"🎉 HOÀN TẤT! Dữ liệu đã được lưu tại: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()