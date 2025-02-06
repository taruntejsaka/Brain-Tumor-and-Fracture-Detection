from ultralytics import YOLO

# Path to your data.yaml file
data_yaml_path = "/Users/sirisipallinarendra/Desktop/main.py/BONE CANCER SEGMENTATION.v1i.yolov11/data.yaml"

# Load a pre-trained YOLOv11 model (or start from scratch)
model = YOLO("yolov11n.pt")  # You can use other variants like yolov11s, yolov11m, etc.

# Train the model
results = model.train(
    data=data_yaml_path,       # Path to your data.yaml file
    epochs=50,                 # Number of training epochs
    imgsz=640,                 # Image size (default is 640x640)
    batch=16,                  # Batch size
    device="0",                # Use GPU (e.g., "0" for GPU 0, or "cpu" for CPU)
    workers=4,                 # Number of workers for data loading
    project="bone_cancer",     # Project name
    name="exp1",               # Experiment name
    save=True,                 # Save checkpoints
    optimizer="auto",          # Optimizer (SGD, Adam, etc.)
    lr0=0.01,                  # Initial learning rate
)

# Evaluate the model on the validation set
metrics = model.val()

# Export the trained model (optional)
model.export(format="onnx")  # Export to ONNX format for deployment