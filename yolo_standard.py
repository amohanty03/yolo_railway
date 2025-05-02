from ultralytics import YOLO
import yaml
import os

def fix_label_files():
    """Fix label files by converting class index 1 to 0"""
    import glob
    from pathlib import Path

    for split in ['train', 'test', 'valid']:
        label_files = glob.glob(f'labels/{split}/*.txt')
        print(f"Processing {len(label_files)} {split} label files...")
        for label_path in label_files:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '1':
                    parts[0] = '0'
                    fixed_lines.append(' '.join(parts) + '\n')
                else:
                    fixed_lines.append(line)
            with open(label_path, 'w') as f:
                f.writelines(fixed_lines)

def create_data_yaml():
    current_dir = os.getcwd()
    data = {
        'path': current_dir,
        'train': os.path.join(current_dir, 'images/train'),
        'val':   os.path.join(current_dir, 'images/valid'),
        'test':  os.path.join(current_dir, 'images/test'),
        'names': ['railway'],
        'nc':    1
    }
    os.makedirs('dataset', exist_ok=True)
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def train_yolo():
    model = YOLO('yolov8n.pt')
    args = {
        'data':         'dataset/data.yaml',
        'epochs':       50,
        'imgsz':        1024,
        'batch':        8,
        'device':       0,
        'workers':      4,
        'patience':     50,
        'project':      'runs/train',
        'name':         'exp1',
        'pretrained':   True,
        'optimizer':    'SGD',
        'lr0':          1e-4,
        'weight_decay': 5e-4,
        'warmup_epochs':   5,
        'warmup_momentum': 0.5,
        'warmup_bias_lr':  0.05,
        'box':  7.5,
        'cls':  0.5,
        'dfl':  1.5,
        'save': True,
        'save_period': 10,
        'cache': False,
        'val':   True,
        'amp':   False,
        'fraction': 1.0,
        'exist_ok': True,
        'seed':  42
    }
    try:
        verify_dataset()
        print("Attempting training with CUDA...")
        results = model.train(**args)
        model.export(format='torchscript')
        model.save('best_model.pt')
        return results
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        if "CUDA" in str(e):
            print("CUDA error, retrying on CPU...")
            args.update(device='cpu', batch=2, workers=2)
            results = model.train(**args)
            model.export(format='torchscript')
            model.save('best_model.pt')
            return results
        raise

def evaluate_on_test_set(model_path='best_model.pt'):
    """
    Evaluate on test images, write out YOLO-format .txt with predicted class 0,
    then compare with ground truth.
    """
    import glob
    from pathlib import Path
    from generate_test_labels import compare_predictions

    model = YOLO(model_path)
    test_images = glob.glob('images/test/*.jpg') + \
                  glob.glob('images/test/*.jpeg') + \
                  glob.glob('images/test/*.png')

    os.makedirs('predictions/test', exist_ok=True)

    total_predictions = 0
    correct_predictions = 0

    for img_path in test_images:
        results = model(img_path)
        img_name = Path(img_path).stem
        pred_path = f'predictions/test/{img_name}.txt'

        with open(pred_path, 'w') as f:
            for r in results:
                for box in r.boxes:
                    cls   = int(box.cls[0])       # 0 for your single-class model
                    conf  = float(box.conf[0])
                    x_c, y_c, w, h = box.xywhn[0].tolist()

                    # write the predicted class (0), not hard-coded 1
                    f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

                    total_predictions += 1
                    if conf > 0.5:
                        correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions else 0
    results = compare_predictions('labels/test', 'predictions/test')

    print("\nTest Set Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("\nDetailed Metrics:")
    for k, v in results['metrics'].items():
        print(f"{k.replace('_',' ').title()}: {v}")
    return {**results, 'accuracy': accuracy}

def verify_dataset():
    import glob
    from pathlib import Path

    image_files = glob.glob('images/train/*.jpg') + \
                  glob.glob('images/train/*.jpeg') + \
                  glob.glob('images/train/*.png')
    print(f"Found {len(image_files)} training images")

    valid_pairs = 0
    empty_labels = 0

    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = f'labels/train/{img_name}.txt'
        if os.path.exists(label_path):
            content = open(label_path).read().strip()
            if content:
                valid_pairs += 1
            else:
                empty_labels += 1
        else:
            print(f"Missing label for {img_name}")

    print(f"Valid pairs: {valid_pairs}, Empty labels: {empty_labels}")
    if valid_pairs == 0:
        raise ValueError("No valid image-label pairs found.")
    return True

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    fix_label_files()
    create_data_yaml()
    cache_file = 'labels/train.cache'
    if os.path.exists(cache_file):
        os.remove(cache_file)

    if not verify_dataset():
        print("Dataset verification failed.")
        exit(1)

    try:
        val_results = train_yolo()
        print("\nValidation Results:\n", val_results)
        test_results = evaluate_on_test_set()
    except Exception as e:
        print(f"Training/Evaluation failed: {e}")
        print("Check dataset and CUDA configuration.")
        exit(1)