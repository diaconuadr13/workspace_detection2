import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the CSV files
file_paths_iou_csv = {
    "A": "run-unet_experiment_AAA-tag-IoU_val.csv",
    "B": "run-unet_experiment_B-tag-IoU_val.csv",
    "C": "run-unet_experiment_CC-tag-IoU_val.csv"
}

data_iou_csv = {}
for key, path in file_paths_iou_csv.items():
    data_iou_csv[key] = pd.read_csv(path, header=None)

# Extract epochs and IoU values from CSV
epochs_iou_csv = list(range(len(data_iou_csv["A"])))
iou_A_csv = pd.to_numeric(data_iou_csv["A"][2].tolist(), errors='coerce')
iou_B_csv = pd.to_numeric(data_iou_csv["B"][2].tolist(), errors='coerce')
iou_C_csv = pd.to_numeric(data_iou_csv["C"][2].tolist(), errors='coerce')

# Smooth the IoU values using a Gaussian filter with sigma=0.67
iou_A_smooth_csv = gaussian_filter1d(iou_A_csv, sigma=0.67)
iou_B_smooth_csv = gaussian_filter1d(iou_B_csv, sigma=0.67)
iou_C_smooth_csv = gaussian_filter1d(iou_C_csv, sigma=0.67)

# Plot the smoothed IoU data without markers and with a larger legend
plt.figure(figsize=(10, 6))
plt.plot(epochs_iou_csv, iou_A_smooth_csv, label='A')
plt.plot(epochs_iou_csv, iou_B_smooth_csv, label='B')
plt.plot(epochs_iou_csv, iou_C_smooth_csv, label='C')

plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Validation IoU over Epochs')
plt.legend(fontsize='large')
plt.grid(True)
plt.show()
