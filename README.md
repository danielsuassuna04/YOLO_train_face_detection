# YOLO Model Training for Face Detection

This repository contains a Python implementation for training a YOLO model on a face detection dataset downloaded from Kaggle. The code fine-tunes a pre-trained YOLO model and evaluates it on the dataset.

## Features
- **Download Dataset from Kaggle**: Automatically downloads and unzips the dataset from Kaggle.
- **Hyperparameter Customization**: Augmentation and optimization parameters are configurable.
- **Model Training**: Fine-tunes the YOLO model for face detection.
- **Evaluation and Visualization**: Evaluates the trained model and visualizes predictions.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8 or higher
- Required Python libraries listed in `requirements.txt`

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/danielsuassuna04/YOLO_train_face_detection.git
cd YOLO_train_face_detection
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset:

The script will automatically download the dataset using `kagglehub`. Ensure you have your Kaggle API key set up. Refer to [Kaggle API documentation](https://www.kaggle.com/docs/api) for setup instructions.

```python
import kagglehub
path = kagglehub.dataset_download("fareselmenshawii/face-detection-dataset")
print("Path to dataset files:", path)
```

4. Train the YOLO model:

The script uses a pre-trained YOLO model and fine-tunes it on the downloaded dataset. Training parameters such as epochs, batch size, and augmentation settings can be adjusted.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

hyp_params = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0
}

model.train(
    data="dataset.yaml",
    epochs=10,
    hsv_h=hyp_params["hsv_h"],
    hsv_s=hyp_params["hsv_s"],
    hsv_v=hyp_params["hsv_v"],
    imgsz=640,
    batch=16,
    name="custom_yolo_training2",
    augment=True,
    pretrained=True,
    optimizer="Adamw",
)
```

5. Evaluate the model:

```python
model.val(data="dataset.yaml")
```

6. Visualize predictions:

The script generates visualizations for predictions made by the trained YOLO model.

```python
import matplotlib.pyplot as plt
import cv2

image_path = "your_image.jpg"
results = model.predict(image_path, conf=0.5)

results[0].plot()  # Save the plotted result
img_with_boxes = results[0].plot(show=False, save=False)

plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

7. Save the trained model:

The trained model is saved to a specified directory for future use.

```python
model.save("/content/drive/MyDrive/models/face_detection_2.pt")
```

## Notes
- Ensure that the dataset is compatible with YOLO's expected format and that the `dataset.yaml` file is correctly configured.
- Use appropriate hardware (e.g., GPU) for faster training and evaluation.
- Modify the training parameters and augmentation settings to achieve optimal performance for your specific dataset.

## License
This project is licensed under the MIT License. Feel free to modify and use it as needed.

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Kaggle](https://www.kaggle.com/)
- [OpenCV](https://opencv.org/)

