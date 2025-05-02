import pandas as pd
import matplotlib.pyplot as plt

# 1. load your training log
df = pd.read_csv('runs/train/exp1/results.csv')

# 2. compute total train & val loss
df['train_loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
df['val_loss']   = df['val/box_loss']   + df['val/cls_loss']   + df['val/dfl_loss']

# 3. pick a “accuracy” metric — for detection we often use mAP@50
df['map50'] = df['metrics/mAP50(B)']

# 4. make the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss curve
ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
ax1.plot(df['epoch'], df['val_loss'],   label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss over Epochs')
ax1.legend()
ax1.grid(True)

# mAP curve
ax2.plot(df['epoch'], df['map50'], label='mAP@50')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mAP@50')
ax2.set_title('Detection mAP over Epochs')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
