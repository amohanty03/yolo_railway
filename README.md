
# Railway Track Detection using YOLOv8 (Standard Dataset)

This project leverages the YOLOv8 object detection framework to detect railway tracks from drone footage. It includes dataset pre-processing, training, inference, and detailed evaluation.

---

## Project Structure

```
.
├── annotations/                  # Original annotation data (if any)
├── dataset/
│   └── data.yaml                 # YOLOv8 dataset config
├── images/                       # Image directory (split by use)
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/                       # YOLO-format labels
│   ├── train/
│   ├── valid/
│   └── test/
├── predictions/                 # Model output predictions
├── runs/
│   └── detect/                  # YOLOv8 inference runs
│   └── train/                   # YOLOv8 training runs
├── YOLOv8/                      # YOLO-specific scripts
│
├── best_model.pt                # Trained YOLOv8 model
├── yolov8n.pt                   # Pretrained YOLOv8n checkpoint
│
├── coco_to_yolo.py              # Convert COCO annotations to YOLO format
├── generate_test_labels.py      # Optional: Generate test labels if needed
├── metrics.py                   # Evaluation script
├── graph.py                     # Plot loss and mAP curves
├── output.py                    # Export or visualization utilities
├── yolo_inference.py            # Run inference and save results
├── yolo_standard.py             # Training script (Standard dataset)
└── Untitled presentation.pdf    # Supporting slides (if any)
```

---

## Requirements

```bash
pip install ultralytics opencv-python matplotlib
```

---

## How to Train

Edit `yolo_standard.py` if needed, then:

```bash
python yolo_standard.py
```

Training uses `yolov8n.pt` pretrained on COCO. Training runs for 50 epochs with:

- Image size: 1024×1024
- Batch size: 8
- Optimizer: SGD with lr=1e-4
- Early stopping: patience 50
- Warm-up and learning rate scheduling enabled

---

## How to Run Inference

```bash
python yolo_inference.py
```

Results will be saved under `predictions/test/labels/`.

---

## Evaluation

```bash
python metrics.py
```

Metrics include:
- Precision / Recall / F1
- True/False Positives/Negatives
- mAP@0.5
