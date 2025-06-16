import cv2
import numpy as np
from PIL import Image
import os

# Path to face image dataset
path = 'dataset'

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to get images and IDs
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        pil_img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_numpy = np.array(pil_img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])

        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
            .detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Save the trained model
os.makedirs('trainer', exist_ok=True)
recognizer.save('trainer/trainer.yml')

print(f"✅ Training complete! Trained on {len(np.unique(ids))} user(s).")

