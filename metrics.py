from ultralytics import YOLO

def run_validation():
    model = YOLO('best_model.pt')
    metrics = model.val(
        data='dataset/data.yaml',
        imgsz=1024,
        batch=8,
        device=0,
        workers=4           # you can leave this >0 if wrapped correctly
    )
    print(metrics)

if __name__ == "__main__":
    # On Windows, if you plan to freeze this script into an exe:
    # from multiprocessing import freeze_support
    # freeze_support()

    run_validation()
