import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog
import os

# Load data from CSV file
# Assuming the CSV has a 'class' column for labels and all other columns are features
def load_data_from_csv(csv_file_path):
    """Load data from CSV file with 'class' column as labels."""
    df = pd.read_csv(csv_file_path)
    
    # Check if 'class' column exists
    if 'class' not in df.columns:
        raise ValueError("CSV file must contain a 'class' column for labels")
    
    # Separate features and labels
    X = df.drop('class', axis=1).values
    y = df['class'].values
    feature_names = df.drop('class', axis=1).columns.tolist()
    
    return X, y, feature_names

def select_csv_file():
    """Open a file dialog to select a CSV file."""
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Set initial directory to 'data' folder
    data_dir = os.path.join(os.getcwd(), 'data')
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir=data_dir
    )
    
    root.destroy()  # Clean up the root window
    return file_path

# Load data using file picker
print("Please select your CSV file...")
csv_file_path = select_csv_file()

if not csv_file_path:
    print("No file selected. Exiting.")
    exit()

try:
    X, y, feature_names = load_data_from_csv(csv_file_path)
    print(f"Successfully loaded data from: {csv_file_path}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    print(f"Classes: {np.unique(y)}")
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Unexpected error: {e}")
    exit()

# Generate unique colors for each class
def generate_colors(n_classes):
    """Generate unique colors for each class."""
    # Use a colormap to generate distinct colors
    cmap = plt.cm.tab10 if n_classes <= 10 else plt.cm.tab20
    colors = [cmap(i) for i in range(n_classes)]
    return colors

# Get unique classes and generate colors
unique_classes = np.unique(y)
n_classes = len(unique_classes)
colors = generate_colors(n_classes)

# Create a mapping from class labels to indices for color selection
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

# Normalize features
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Use LDA to sort features by absolute coefficient importance
lda = LinearDiscriminantAnalysis()
lda.fit(X_norm, y)

# For multiclass, sum absolute values across all discriminant axes
lda_abs = np.abs(lda.coef_)
if lda_abs.ndim == 2:
    lda_importance = lda_abs.sum(axis=0)
else:
    lda_importance = lda_abs
sorted_indices = np.argsort(-lda_importance)  # descending order

# Reorder data and feature names
X_norm_sorted = X_norm[:, sorted_indices]
feature_names_sorted = [feature_names[i] for i in sorted_indices]

# Parameters
n_features = X.shape[1]
angles = np.linspace(0, np.pi/2, n_features)  # 0 to 90 degrees

def glc_l_encoding(x):
    """General Line Coordinates (length encoding)."""
    return np.cumsum([[x[i] * np.cos(angles[i]), x[i] * np.sin(angles[i])] for i in range(n_features)], axis=0)

def glxac_l_encoding(x):
    """GLxAC Coordinates (angle encoding with unit length)."""
    return np.cumsum([[np.cos(x[i]*np.pi/2), np.sin(x[i]*np.pi/2)] for i in range(n_features)], axis=0)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
titles = ['GLC-L (Length Encoding)', 'GL×AC-L (Angle Encoding)']

for idx, encoding_func in enumerate([glc_l_encoding, glxac_l_encoding]):
    ax = axes[idx]
    for class_label in unique_classes:
        X_class = X_norm_sorted[y == class_label]
        color_idx = class_to_index[class_label]
        for row in X_class[:10]:  # Plot first 10 samples per class for clarity
            path = encoding_func(row)
            ax.plot(path[:, 0], path[:, 1], alpha=0.6, color=colors[color_idx], label=class_label if row is X_class[0] else "")
    ax.set_title(titles[idx] + "\n(Features sorted by LDA importance)")
    ax.axis('equal')
    ax.grid(True)
    # Add legend to show class labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.suptitle(f"CSV Data Visualization in Two Encoding Schemes\n(Features sorted by LDA importance, {n_classes} classes)")
plt.tight_layout()
plt.show()