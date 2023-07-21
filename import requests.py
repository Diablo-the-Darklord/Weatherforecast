import cv2
import albumentations as A
from matplotlib import pyplot as plt

# Sample image path
image_path = "C:\Users\aksha\OneDrive\Pictures\Wallpaper\4k-blue-goku-2021-dragon-ball-anime-wallpaper-11628961535wjbiwuvknh.jpg"

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (Albumentations expects RGB)

# Define a list of augmentations
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
])

# Apply the augmentations
augmented = transform(image=image)
augmented_image = augmented["image"]

# Display the original and augmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title("Augmented Image")
plt.axis("off")

plt.show()
