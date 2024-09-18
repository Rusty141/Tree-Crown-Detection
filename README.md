# Tree Crown Detection Code using DeepForest

## Overview
This code detects tree crowns in images using a pre-trained DeepForest model. It performs predictions through sliding windows and tile-based approaches, visualizing results with bounding boxes and circles around detected crowns.

---

## 1. Importing Libraries

```python
from deepforest import main, preprocess
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
```

- **deepforest:** Provides the pre-trained tree detection model and window processing.
- **torch:** Manages computation on the CPU/GPU.
- **cv2 (OpenCV):** For drawing shapes and handling image processing.
- **matplotlib:** Displays images and plots.
- **numpy:** Manages numerical operations, especially on image arrays.
- **PIL (Pillow):** Loads images in various formats.
- **pandas:** For handling prediction results.

---

## 2. Loading and Configuring the DeepForest Model

```python
model = main.deepforest()
model.use_release()  # Use the pre-trained model
model.current_device = torch.device("cpu")  # Set the model to use CPU
```

- The pre-trained DeepForest model is loaded for tree crown detection.
- The model is set to run on the CPU.

---

## 3. Image Upload and Conversion

```python
def load_image(image_path):
    return Image.open(image_path)
    
image_path = r"C:\path\to\image.jpg"
image = load_image(image_path)

image_np = np.array(image).astype('float32')
```

- The function `load_image` loads an image from the local path using PIL.
- The image is converted to a NumPy array for further processing.

---

## 4. Sliding Window Predictions

```python
windows = preprocess.compute_windows(image_np, patch_size=400, patch_overlap=0)
```

- The image is divided into non-overlapping 400x400 pixel windows for localized predictions.

```python
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
for index2 in range(min(4, len(windows))):
    crop = image_np[windows[index2].indices()]
    boxes = model.predict_image(image=np.flip(crop[..., ::-1], 2), return_plot=True)
    axes[index2].imshow(boxes[..., ::-1])
plt.show()
```

- The first four windows are predicted for tree crowns.
- Bounding boxes are visualized for each prediction.

---

## 5. Full Image Prediction with Circles

```python
pred_boxes = model.predict_image(image=image_np)
for index, row in pred_boxes.iterrows():
    center_x = int((row["xmin"] + row["xmax"]) / 2)
    center_y = int((row["ymin"] + row["ymax"]) / 2)
    radius = int(min(row["xmax"] - row["xmin"], row["ymax"] - row["ymin"]) / 2)
    cv2.circle(image_with_circles, (center_x, center_y), radius, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
```

- Full image prediction is performed, and circles are drawn around detected crowns.
- The center and radius of each bounding box are computed to plot circles for better visualization.

---

## 6. Tile-Based Predictions with Bounding Boxes

```python
tile = model.predict_tile(image=image_np, patch_overlap=0, iou_threshold=0.05, patch_size=400)
for index, row in tile.iterrows():
    cv2.rectangle(image_tile, (int(row["xmin"]), int(row["ymin"]), (int(row["xmax"]), int(row["ymax"])), (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)
```

- The image is split into tiles for enhanced accuracy over large areas.
- Bounding boxes are drawn around detected objects within the tile.

---

## 7. Visualization of Results

```python
fig = plt.figure(figsize=(15, 15))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(np.flip(image_np, 2))

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(np.flip(image_tile, 2))
plt.show()
```

- Displays the original image alongside the image with tile-based predictions for comparison.

---

## Conclusion
This code effectively detects tree crowns using both window-based and tile-based approaches, providing a comprehensive method for visualizing the results through bounding boxes and circles on the image.

---

## Reference

For more information on using DeepForest for tree crown detection, you can refer to the [Environmental AI Book - DeepForest Tree Crown Detection](https://acocac.github.io/environmental-ai-book/forest/modelling/forest-modelling-treecrown_deepforest.html).
