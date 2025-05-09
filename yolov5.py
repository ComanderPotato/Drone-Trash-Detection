from pathlib import Path
from ultralytics import YOLO
data_path = str(Path('./data/taco.yaml').resolve())
print(data_path)
model = YOLO('yolov5s.pt')  # or 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'

# Train on your dataset (e.g., COCO-style YAML format)
# model.train(data=data_path, epochs=50, imgsz=640)
model.predict('data/test/00035.jpg')
# results = model(source='data/test/00035.jpg')
# # results['boxes']
# print(results)
# results.print()
# results.show()
# results.print()

# import torch

# # Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

# # Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
# img = "Lamborghini-Huracan-Yellow-Sydney-Car-Hire-01.jpg"
# # Perform inference (handles batching, resizing, normalization automatically)
# results = model(img)

# # Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
# results.print()  # Print results to console
# results.show()  # Display results in a window
# results.save()  # Save results to runs/detect/exp
