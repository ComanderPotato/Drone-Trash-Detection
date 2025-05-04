## Install

1. Activate virtual environment
2. In terminal, pip install -r requirements.txt

## Installed libraries

- TensorFlow
- Optuna (hyperparameter optimisation)
- Matplotlib (visualisation)
- Pillow (For images)
- Scikit-learn
- Opencv
- ultralytics (Yolo architecture)
- segmentation-models

## Pipeline

### Data Preparation

- Data collection
- Data preprocessing (normalisation, standardisation)
- Data augmentation
- splitting data - train/test/validation 80-10-10, 70-15-15

### Feature engineering

- HOG descriptors
- Data augmentation

### Modeling & Experimenting

- Transfer learning
- Model selection (Yolov5, Unet/Resnet, Mask-RCNN, Faster-RCNN)
- Model implementation (ultralytics, tensorflow-hub)
- Model architecture (v5/6/7, .., .., Resnet50/Resnet101/Resnet152)
- Compile options (optimizer, loss, metrics)
- Early stopping criteria (callbacks - tensorflow)
- training epochs > 1000

### Evaluation

- Intersection over Union (IoU)
- Accuracy, recall, confusion matrix, precision, F1 score
- K-fold cross validation (overfitting)
- Overfitting & underfitting
