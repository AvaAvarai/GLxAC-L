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
    """GLC-L: angles from arccos(|k_i|), lengths from x_i, cumulative sum of vectors"""
    # Get normalized coefficients (k_i) - these are the LDA coefficients
    k_i = lda_importance  # Already sorted in decreasing order from earlier
    
    # Normalize coefficients to [-1, 1] range as per PDF
    k_max = np.max(np.abs(k_i))
    k_normalized = k_i / k_max  # This gives k_i in [-1, 1] range
    
    # Calculate angles: Q_i = arccos(|k_i|)
    angles_from_k = np.arccos(np.abs(k_normalized))
    
    # Calculate vectors: [x_i * cos(Q_i), x_i * sin(Q_i)]
    glc_vectors = []
    for i in range(n_features):
        # x_i is the feature value (length)
        # Q_i = arccos(|k_i|) is the angle
        angle = angles_from_k[i]
        length = x[i]  # Feature value determines segment length
        vector = [length * np.cos(angle), length * np.sin(angle)]
        glc_vectors.append(vector)
    
    return np.cumsum(glc_vectors, axis=0)

def glxac_l_encoding(x):
    """GLCxA-L: cumsum(k_i * q_i) where k_i are LDA coefficients, q_i = arccos(|x_i|)"""
    # Get LDA coefficients in decreasing order
    k_i = lda_importance  # Already sorted in decreasing order from earlier
    
    # x_i are already min-max normalized from earlier preprocessing
    # Calculate q_i = arccos(|x_i|)
    q_i = np.arccos(np.abs(x))
    
    # Calculate k_i * q_i for each feature
    kq_vectors = []
    for i in range(n_features):
        # Create a unit vector in the direction of the LDA coefficient
        # Use the angle from the original angles array (0 to 90 degrees)
        angle = angles[i]  # This is the fixed angle for feature i
        # Scale by k_i * q_i
        magnitude = k_i[i] * q_i[i]
        vector = [magnitude * np.cos(angle), magnitude * np.sin(angle)]
        kq_vectors.append(vector)
    
    return np.cumsum(kq_vectors, axis=0)

def find_optimal_separation_and_accuracy(final_endpoints, unique_classes):
    """Find optimal separation point on U-axis and calculate classification accuracy."""
    # Extract x-coordinates (U-axis positions) and class labels - vectorized
    u_positions = np.array([endpoint[0] for endpoint, color, class_label in final_endpoints])
    class_labels = np.array([class_label for endpoint, color, class_label in final_endpoints])
    
    # Find optimal separation point
    if len(unique_classes) == 2:
        # Binary case: vectorized operations
        first_class_mask = (class_labels == unique_classes[0])
        first_class_positions = u_positions[first_class_mask]
        other_class_positions = u_positions[~first_class_mask]
        
        threshold = (np.mean(first_class_positions) + np.mean(other_class_positions)) / 2
        
        # Vectorized accuracy calculation
        predictions = (u_positions >= threshold).astype(int)
        true_labels = (class_labels == unique_classes[1]).astype(int)
        accuracy = np.mean(predictions == true_labels)
        
    else:
        # Multi-class case: vectorized
        threshold = np.mean(u_positions)
        
        # Vectorized class means calculation
        class_means = {}
        for class_label in unique_classes:
            class_mask = (class_labels == class_label)
            if np.any(class_mask):
                class_means[class_label] = np.mean(u_positions[class_mask])
        
        # Vectorized prediction
        predictions = []
        for pos in u_positions:
            closest_class = min(class_means.keys(), 
                              key=lambda c: abs(pos - class_means[c]))
            predictions.append(closest_class)
        
        accuracy = np.mean(np.array(predictions) == class_labels)
    
    return threshold, accuracy

def plot_with_shared_u_axis(ax, encoding_func, X_data, y_data, colors, class_to_index, unique_classes):
    """Plot paths with shared U-axis, first class up, others down, with endpoint projections."""
    # Pre-allocate arrays for better performance
    all_segments = []
    final_endpoints = []
    
    # Vectorized path generation for all classes at once
    all_paths = []
    all_colors = []
    all_class_labels = []
    all_y_directions = []
    
    for class_idx, class_label in enumerate(unique_classes):
        X_class = X_data[y_data == class_label]
        color_idx = class_to_index[class_label]
        y_direction = 1 if class_idx == 0 else -1
        
        # Generate all paths for this class at once
        for row in X_class:
            path = encoding_func(row)
            # Apply y-direction transformation
            transformed_path = path.copy()
            transformed_path[:, 1] = transformed_path[:, 1] * y_direction
            
            all_paths.append(transformed_path)
            all_colors.append(colors[color_idx])
            all_class_labels.append(class_label)
            all_y_directions.append(y_direction)
    
    # Batch plot first segments
    for i, path in enumerate(all_paths):
        ax.plot([0, path[0][0]], [0, path[0][1]], 
               color=all_colors[i], linewidth=2, alpha=0.7, zorder=2)
    
    # Collect all segments efficiently
    for i, path in enumerate(all_paths):
        class_label = all_class_labels[i]
        color = all_colors[i]
        
        # Store segments for overlap detection
        for j in range(len(path) - 1):
            segment = {
                'start': path[j],
                'end': path[j + 1],
                'color': color,
                'class_label': class_label,
                'linewidth': 1.0,
                'alpha': 0.6
            }
            all_segments.append(segment)
        
        # Store final endpoint for projection
        final_endpoints.append((path[-1], color, class_label))
    
    # Optimized overlap detection using spatial indexing
    if len(all_segments) > 0:
        # Group segments by class for faster overlap detection
        segments_by_class = {}
        for segment in all_segments:
            class_label = segment['class_label']
            if class_label not in segments_by_class:
                segments_by_class[class_label] = []
            segments_by_class[class_label].append(segment)
        
        # Process overlaps by class (only same class overlaps)
        for class_label, class_segments in segments_by_class.items():
            if len(class_segments) > 1:
                # Vectorized midpoint calculation
                midpoints = np.array([(seg['start'] + seg['end']) / 2 for seg in class_segments])
                
                # Efficient overlap detection using broadcasting
                for i, seg1 in enumerate(class_segments):
                    overlap_count = 1
                    # Calculate distances to all other segments efficiently
                    distances = np.linalg.norm(midpoints - midpoints[i], axis=1)
                    overlap_count += np.sum((distances < 0.05) & (np.arange(len(distances)) != i))
                    
                    # Adjust linewidth based on overlap count
                    seg1['linewidth'] = min(1.0 + (overlap_count - 1) * 0.5, 5.0)
    
    # Batch plot all segments
    for segment in all_segments:
        ax.plot([segment['start'][0], segment['end'][0]], 
                [segment['start'][1], segment['end'][1]], 
                color=segment['color'], 
                linewidth=segment['linewidth'],
                alpha=segment['alpha'])
    
    # Batch project final endpoints to U-axis
    for endpoint, color, class_label in final_endpoints:
        x_end, y_end = endpoint
        # Draw dotted line from endpoint to U-axis
        ax.plot([x_end, x_end], [y_end, 0], 
               color=color, linestyle=':', alpha=0.6, linewidth=1)
        # Draw colored dots
        ax.scatter(x_end, y_end, color=color, s=8, alpha=0.6, zorder=10, 
                  edgecolor='black', linewidth=0.5)
        ax.scatter(x_end, 0, color=color, s=6, alpha=0.6, zorder=10, 
                  edgecolor='black', linewidth=0.5)
    
    # Find optimal separation and draw separation line
    threshold, accuracy = find_optimal_separation_and_accuracy(final_endpoints, unique_classes)
    
    # Draw separation line
    ax.axvline(x=threshold, color='yellow', linestyle='--', linewidth=1, alpha=0.8, zorder=5)
    
    # Display accuracy
    accuracy_text = f"Accuracy: {accuracy:.3f}"
    ax.text(0.02, 0.98, accuracy_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    # Draw shared U-axis
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8, zorder=1)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
titles = ['GLC-L (Length Encoding)', 'GL×AC-L (Angle Encoding)']

# Create legend handles for all classes
legend_handles = []
for i, class_label in enumerate(unique_classes):
    color_idx = class_to_index[class_label]
    handle = plt.Line2D([], [], color=colors[color_idx], linewidth=2, label=str(class_label))
    legend_handles.append(handle)

for idx, encoding_func in enumerate([glc_l_encoding, glxac_l_encoding]):
    ax = axes[idx]
    plot_with_shared_u_axis(ax, encoding_func, X_norm_sorted, y, colors, class_to_index, unique_classes)
    ax.set_title(titles[idx] + "\n(Features sorted by LDA importance)")
    ax.axis('equal')
    ax.grid(True)
    ax.set_facecolor('lightgrey')  # Set light grey background

# Add single legend to the figure (not individual subplots)
fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.suptitle(f"CSV Data Visualization in Two Encoding Schemes\n(Features sorted by LDA importance, {n_classes} classes)")
plt.tight_layout()
plt.show()