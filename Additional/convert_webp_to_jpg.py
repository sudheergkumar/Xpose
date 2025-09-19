from PIL import Image
import os

base_path = 'model'  # change if your dataset folder has a different name
classes = ['real', 'edited', 'ai_generated']

for category in classes:
    folder = os.path.join(base_path, category)
    for filename in os.listdir(folder):
        if filename.endswith('.webp'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            new_name = filename.replace('.webp', '.jpg')
            img.save(os.path.join(folder, new_name))
            os.remove(img_path)  # delete old webp
print("Conversion done ✅")
