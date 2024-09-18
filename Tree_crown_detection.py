from deepforest import main, preprocess
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

model = main.deepforest()
model.use_release()
model.current_device = torch.device("cpu")

def load_image(image_path):
    return Image.open(image_path)

image_path = r"C:\path\to\image.jpg"
image = load_image(image_path)
image_np = np.array(image).astype('float32')

windows = preprocess.compute_windows(image_np, patch_size=400, patch_overlap=0)
print(f"We have {len(windows)} windows in the image")

num_windows_to_display = min(4, len(windows))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
axes = axes.flatten()

for index2 in range(num_windows_to_display):
    crop = image_np[windows[index2].indices()]
    boxes = model.predict_image(image=np.flip(crop[..., ::-1], 2), return_plot=True)
    axes[index2].imshow(boxes[..., ::-1])

for index2 in range(num_windows_to_display, len(axes)):
    axes[index2].axis('off')

plt.show()

pred_boxes = model.predict_image(image=image_np)
image_with_circles = image_np.copy()

for index, row in pred_boxes.iterrows():
    center_x = int((row["xmin"] + row["xmax"]) / 2)
    center_y = int((row["ymin"] + row["ymax"]) / 2)
    radius = int(min(row["xmax"] - row["xmin"], row["ymax"] - row["ymin"]) / 2)
    cv2.circle(image_with_circles, (center_x, center_y), radius, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

plt.figure(figsize=(15, 15))
plt.imshow(np.flip(image_with_circles, 2))
plt.axis('off')
plt.show()

tile = model.predict_tile(image=image_np, return_plot=False, patch_overlap=0, iou_threshold=0.05, patch_size=400)
image_tile = image_np.copy()

for index, row in tile.iterrows():
    cv2.rectangle(image_tile, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), 
                  (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)

fig = plt.figure(figsize=(15, 15))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(np.flip(image_np, 2))

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(np.flip(image_tile, 2))

plt.show()
