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
def generate_colors(unique_classes):
    """Generate unique colors for each class with special handling for benign/malignant."""
    n_classes = len(unique_classes)
    colors = []
    
    for i, class_label in enumerate(unique_classes):
        class_lower = str(class_label).lower()
        
        # Special colors for medical classes
        if class_lower == 'benign':
            colors.append('green')
        elif class_lower == 'malignant':
            colors.append('red')
        else:
            # Use colormap for other classes - properly scale by number of classes
            cmap = plt.cm.viridis  # Use a continuous colormap
            # Scale the color index to [0, 1] range based on total number of classes
            color_index = i / (n_classes - 1) if n_classes > 1 else 0
            colors.append(cmap(color_index))
    
    return colors

# Get unique classes and generate colors
unique_classes = np.unique(y)
n_classes = len(unique_classes)
colors = generate_colors(unique_classes)

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

def plot_with_overlap_detection(ax, encoding_func, X_data, y_data, colors, class_to_index, unique_classes):
    """Plot paths with thickened overlapping segments."""
    # Store all segments for overlap detection
    all_segments = []
    
    # First pass: collect all segments
    for class_label in unique_classes:
        X_class = X_data[y_data == class_label]
        color_idx = class_to_index[class_label]
        for row in X_class[:10]:  # Plot first 10 samples per class for clarity
            path = encoding_func(row)
            for i in range(len(path) - 1):
                segment = {
                    'start': path[i],
                    'end': path[i + 1],
                    'color': colors[color_idx],
                    'class_label': class_label,
                    'linewidth': 1.0,
                    'alpha': 0.6
                }
                all_segments.append(segment)
    
    # Second pass: detect overlaps and adjust linewidth
    for i, seg1 in enumerate(all_segments):
        overlap_count = 1  # Start with 1 for the segment itself
        
        for j, seg2 in enumerate(all_segments):
            if i != j:
                # Check if segments overlap (simplified overlap detection)
                # Calculate distance between segment midpoints
                mid1 = (seg1['start'] + seg1['end']) / 2
                mid2 = (seg2['start'] + seg2['end']) / 2
                distance = np.linalg.norm(mid1 - mid2)
                
                # If segments are very close, consider them overlapping
                if distance < 0.05:  # Threshold for overlap detection
                    overlap_count += 1
        
        # Adjust linewidth based on overlap count
        seg1['linewidth'] = min(1.0 + (overlap_count - 1) * 0.5, 5.0)  # Cap at 5.0
    
    # Third pass: plot all segments with adjusted linewidths
    for segment in all_segments:
        ax.plot([segment['start'][0], segment['end'][0]], 
                [segment['start'][1], segment['end'][1]], 
                color=segment['color'], 
                linewidth=segment['linewidth'],
                alpha=segment['alpha'])

# Create legend handles for all classes
legend_handles = []
for i, class_label in enumerate(unique_classes):
    color_idx = class_to_index[class_label]
    handle = plt.Line2D([], [], color=colors[color_idx], linewidth=2, label=str(class_label))
    legend_handles.append(handle)

for idx, encoding_func in enumerate([glc_l_encoding, glxac_l_encoding]):
    ax = axes[idx]
    plot_with_overlap_detection(ax, encoding_func, X_norm_sorted, y, colors, class_to_index, unique_classes)
    ax.set_title(titles[idx] + "\n(Features sorted by LDA importance)")
    ax.axis('equal')
    ax.grid(True)

# Add single legend to the figure (not individual subplots)
fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.suptitle(f"CSV Data Visualization in Two Encoding Schemes\n(Features sorted by LDA importance, {n_classes} classes)")
plt.tight_layout()
plt.show()