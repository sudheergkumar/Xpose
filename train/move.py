import os
import shutil

lfw_root = "C:/Users/gsudh/Downloads/archive/lfw-deepfunneled/lfw-deepfunneled"
real_output = "C:/Users/gsudh/Desktop/Xpose/dataset/real"



os.makedirs(real_output, exist_ok=True)

for person_folder in os.listdir(lfw_root):
    person_path = os.path.join(lfw_root, person_folder)
    if os.path.isdir(person_path):
        for image_file in os.listdir(person_path):
            src = os.path.join(person_path, image_file)
            dst = os.path.join(real_output, f"{person_folder}_{image_file}")
            shutil.copy2(src, dst)  # Use .move(src, dst) if you want to move instead
