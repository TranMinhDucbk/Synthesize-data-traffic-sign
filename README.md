Traffic Light Detection and OCR (code and detail in dataset folder)

Duc Tran



Create synthetic data for training (data for training YOLO model (object detection))

Step 1: Preparing 



Create folder including a smaller folder with picture of the object that is needed to be detected , and that picture has to be removed background and captured in every angle(if possible), and a folder with background. For example, with detecting sign.





open the folder and add smaller folders



Inside the background folder, collect picture from online or take picture of real life

Background picture



The remaining Folder includes object + label(in this situation : sign)





Step 2 : Code by python 

Using python to generate picture ( you can select from 2 options including test file and train file) for yolov11 

Import libraries

Create parameter including : blur, brightness, rotation angle, split, number of images

Create functions including :auto remove bg, clear edges, rotation, augmentation ( blur, bright, rotation), overlay, managing dataset.

Main : create picture in folder, creat yaml file.

Settings:

func1 : choose priorities class

PRIORITY_CLASSES = ['bien_can_doc_theo_cot', 'bien_can_doc_theo_hang', 'bien_can_doc_1_o' , 'bien_can_doc_theo_hang_khong_mui_ten', 'bien_1_mui_ten_1_o']
PRIORITY_RATIO = 0.15

func 2: create number
NUM_IMAGES_TO_GENERATE = 10000     # Tổng số ảnh muốn tạo
TRAIN_VAL_SPLIT_RATIO = 0.8     # 80% Train, 20% Val
MIN_OBJECTS_PER_IMAGE = 2
MAX_OBJECTS_PER_IMAGE = 5
MIN_SCALE = 0.05                # 5% chiều cao ảnh nền
MAX_SCALE = 0.2                 # 20% chiều cao ảnh nền

func3: augmentation
OBJECT_AUG_CHANCE = 0.7         
MAX_BLUR_KERNEL = 5            
MIN_BRIGHTNESS = 0.7
MAX_BRIGHTNESS = 1.3
MAX_ROTATION_ANGLE = 15         
MIN_CONTRAST = 0.5  # 0.5 là giảm tương phản (ảnh nhờ nhờ, sương mù)
MAX_CONTRAST = 1.5  # 1.5 là tăng tương phản (ảnh gắt)

func4: draw pic
CLEAN_EDGES = True              
ERODE_ITERS = 1                 



Link code: download to see more (or you can see it in gg drive)

https://drive.google.com/file/d/1kfGQksJQsFioylAkDy-x-L6mSO68lHnb/view?usp=sharing

TARGET



When using this code you have to change the link to your folder , and change the parameter to meet your requirement. In the picture is the thing you have to change to fit with your device:

Here is my folder structure:



File yaml creating here is suitable for all yolo versions



Step 3 : Training online or offline.



NOTE: if you use code generatevideo you have to change your background from picture (.jpg, .png,..) to video (.mp4, .avi,...

