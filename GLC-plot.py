import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import seaborn as sns

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
    
    # Set initial directory to 'datasets' folder
    data_dir = os.path.join(os.getcwd(), 'datasets')
    # Create datasets directory if it doesn't exist
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

def select_two_classes(unique_classes):
    """Allow user to select which two classes to use for binary classification."""
    root = tk.Tk()
    root.title("Select Two Classes")
    root.geometry("400x300")
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Create and pack widgets
    tk.Label(root, text="Select two classes for binary classification:", font=("Arial", 12)).pack(pady=10)
    
    # Create listbox with all classes
    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=10, font=("Arial", 10))
    listbox.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
    
    # Populate listbox with classes
    for i, class_label in enumerate(unique_classes):
        listbox.insert(tk.END, f"{i+1}. {class_label}")
    
    selected_classes = []
    
    def on_ok():
        nonlocal selected_classes
        selection = listbox.curselection()
        if len(selection) != 2:
            messagebox.showerror("Error", "Please select exactly 2 classes.")
            return
        
        selected_classes = [unique_classes[i] for i in selection]
        root.destroy()
    
    def on_cancel():
        root.destroy()
    
    # Create buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
    
    # Start the GUI
    root.mainloop()
    
    return selected_classes

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
    
    # Check if we have more than 2 classes
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
        print(f"\nDataset has {len(unique_classes)} classes. Please select 2 classes for binary classification.")
        selected_classes = select_two_classes(unique_classes)
        
        if not selected_classes or len(selected_classes) != 2:
            print("No classes selected or incorrect number of classes. Exiting.")
            exit()
        
        # Filter data to only include selected classes
        mask = np.isin(y, selected_classes)
        X = X[mask]
        y = y[mask]
        
        print(f"Selected classes: {selected_classes}")
        print(f"Filtered dataset: {len(X)} samples")
    
    # Ensure we have exactly 2 classes
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        print("Error: Dataset must have exactly 2 classes for binary classification.")
        exit()
        
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

# For binary classification, use the single discriminant axis
lda_importance = np.abs(lda.coef_[0])
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

def find_optimal_separation_and_accuracy(final_endpoints, unique_classes, custom_threshold=None):
    """Find optimal separation point on U-axis and calculate classification accuracy."""
    u_positions = np.array([endpoint[0] for endpoint, color, class_label in final_endpoints])
    class_labels = np.array([class_label for endpoint, color, class_label in final_endpoints])
    
    # Use custom threshold if provided, otherwise calculate optimal threshold
    if custom_threshold is not None:
        threshold = custom_threshold
    else:
        # Find optimal separation point for binary classification
        first_class_mask = (class_labels == unique_classes[0])
        first_class_positions = u_positions[first_class_mask]
        other_class_positions = u_positions[~first_class_mask]
        threshold = (np.mean(first_class_positions) + np.mean(other_class_positions)) / 2
    
    # Calculate accuracy for binary classification: try both possible class assignments
    first_class_mask = (class_labels == unique_classes[0])
    first_class_positions = u_positions[first_class_mask]
    other_class_positions = u_positions[~first_class_mask]
    
    # Try both assignments: class A on left vs class A on right
    best_accuracy = 0.0
    best_assignment = None
    
    # Assignment 1: First class on left, second class on right
    correct = 0
    for i, pos in enumerate(u_positions):
        actual_class = class_labels[i]
        if pos < threshold:
            predicted_class = unique_classes[0]  # First class on left
        else:
            predicted_class = unique_classes[1]  # Second class on right
        
        if actual_class == predicted_class:
            correct += 1
    
    accuracy1 = correct / len(u_positions)
    if accuracy1 > best_accuracy:
        best_accuracy = accuracy1
        best_assignment = (unique_classes[0], unique_classes[1])  # (left_class, right_class)
    
    # Assignment 2: First class on right, second class on left
    correct = 0
    for i, pos in enumerate(u_positions):
        actual_class = class_labels[i]
        if pos < threshold:
            predicted_class = unique_classes[1]  # Second class on left
        else:
            predicted_class = unique_classes[0]  # First class on right
        
        if actual_class == predicted_class:
            correct += 1
    
    accuracy2 = correct / len(u_positions)
    if accuracy2 > best_accuracy:
        best_accuracy = accuracy2
        best_assignment = (unique_classes[1], unique_classes[0])  # (left_class, right_class)
    
    accuracy = best_accuracy
    
    return threshold, accuracy

def generate_predictions_and_confusion_matrix(encoding_func, X_data, y_data, unique_classes, custom_threshold=None):
    """Generate predictions using the encoding function and return confusion matrix."""
    # Collect endpoints for all samples
    final_endpoints = []
    for i, row in enumerate(X_data):
        path = encoding_func(row)
        final_endpoints.append((path[-1], None, y_data[i]))  # (endpoint, color, class_label)
    
    # Get u_positions and find threshold
    u_positions = np.array([endpoint[0] for endpoint, color, class_label in final_endpoints])
    class_labels = np.array([class_label for endpoint, color, class_label in final_endpoints])
    
    # Find optimal threshold for binary classification
    if custom_threshold is not None:
        threshold = custom_threshold
    else:
        first_class_mask = (class_labels == unique_classes[0])
        first_class_positions = u_positions[first_class_mask]
        other_class_positions = u_positions[~first_class_mask]
        threshold = (np.mean(first_class_positions) + np.mean(other_class_positions)) / 2
    
    # Generate predictions for binary classification: try both assignments and choose the better one
    first_class_mask = (class_labels == unique_classes[0])
    first_class_positions = u_positions[first_class_mask]
    other_class_positions = u_positions[~first_class_mask]
    
    # Try both assignments
    correct1 = 0
    correct2 = 0
    pred1 = []
    pred2 = []
    
    for i, pos in enumerate(u_positions):
        actual_class = class_labels[i]
        
        # Assignment 1: First class on left, second class on right
        if pos < threshold:
            predicted_class1 = unique_classes[0]
        else:
            predicted_class1 = unique_classes[1]
        pred1.append(predicted_class1)
        if actual_class == predicted_class1:
            correct1 += 1
        
        # Assignment 2: First class on right, second class on left
        if pos < threshold:
            predicted_class2 = unique_classes[1]
        else:
            predicted_class2 = unique_classes[0]
        pred2.append(predicted_class2)
        if actual_class == predicted_class2:
            correct2 += 1
    
    # Choose the better assignment
    if correct1 >= correct2:
        predictions = pred1
    else:
        predictions = pred2
    
    # Generate confusion matrix
    cm = confusion_matrix(class_labels, predictions, labels=unique_classes)
    
    return predictions, cm, threshold

def optimize_gac_l_scaling(X_data, y_data, unique_classes, class_to_index, colors):
    """Find the optimal scaling factor h and threshold for GAC-L by maximizing classification accuracy."""
    best_h = 1.0
    best_threshold = None
    best_acc = 0.0
    h_values = np.linspace(0.01, 5.0, 100)  # Coarser search for h
    
    for h in h_values:
        def scaled_encoding(x):
            return np.cumsum([[h * np.cos(x[i] * np.pi / 2), h * np.sin(x[i] * np.pi / 2)] for i in range(len(x))], axis=0)
        
        # Collect endpoints
        final_endpoints = []
        for class_label in unique_classes:
            X_class = X_data[y_data == class_label]
            color_idx = class_to_index[class_label]
            for row in X_class:
                path = scaled_encoding(row)
                final_endpoints.append((path[-1], colors[color_idx], class_label))
        
        # Get u_positions for threshold optimization
        u_positions = np.array([endpoint[0] for endpoint, color, class_label in final_endpoints])
        class_labels = np.array([class_label for endpoint, color, class_label in final_endpoints])
        
        # Try different threshold positions for binary classification
        first_class_mask = (class_labels == unique_classes[0])
        first_class_positions = u_positions[first_class_mask]
        other_class_positions = u_positions[~first_class_mask]
        
        if len(first_class_positions) > 0 and len(other_class_positions) > 0:
            min_pos = min(np.min(first_class_positions), np.min(other_class_positions))
            max_pos = max(np.max(first_class_positions), np.max(other_class_positions))
            threshold_candidates = np.linspace(min_pos, max_pos, 50)
        else:
            threshold_candidates = [np.mean(u_positions)]
        
        # Find best threshold for this h value
        best_threshold_for_h = None
        best_acc_for_h = 0.0
        
        for threshold in threshold_candidates:
            _, acc = find_optimal_separation_and_accuracy(final_endpoints, unique_classes, threshold)
            if acc > best_acc_for_h:
                best_acc_for_h = acc
                best_threshold_for_h = threshold
        
        # Update global best if this h gives better accuracy
        if best_acc_for_h > best_acc:
            best_acc = best_acc_for_h
            best_h = h
            best_threshold = best_threshold_for_h
    
    return best_h, best_threshold, best_acc

def gac_l_encoding(x, h=1.0):
    # x[i] in [0,1], angle = x[i] * 90 degrees = x[i] * (π/2) radians, scaled by h
    return np.cumsum([[h * np.cos(x[i] * np.pi / 2), h * np.sin(x[i] * np.pi / 2)] for i in range(n_features)], axis=0)

# Find optimal h for GAC-L
best_h, best_threshold, best_acc = optimize_gac_l_scaling(X_norm_sorted, y, unique_classes, class_to_index, colors)
print(f'Optimal scaling factor for GAC-L: h={best_h:.3f}, threshold={best_threshold:.3f} (accuracy={best_acc:.3f})')

def plot_with_shared_u_axis(ax, encoding_func, X_data, y_data, colors, class_to_index, unique_classes, custom_threshold=None):
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
    
    # Collect all segments efficiently (including first segments)
    for i, path in enumerate(all_paths):
        class_label = all_class_labels[i]
        color = all_colors[i]
        
        # Store segments for overlap detection (including first segment from origin)
        for j in range(len(path)):
            if j == 0:
                # First segment from origin to first point
                segment = {
                    'start': np.array([0, 0]),
                    'end': path[j],
                    'color': color,
                    'class_label': class_label,
                    'linewidth': 0.5,  # Extra thin by default
                    'alpha': 0.6
                }
            else:
                # Subsequent segments
                segment = {
                    'start': path[j-1],
                    'end': path[j],
                    'color': color,
                    'class_label': class_label,
                    'linewidth': 0.5,  # Extra thin by default
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
                    seg1['linewidth'] = min(0.5 + (overlap_count - 1) * 0.25, 2.0)  # Extra thin, cap at 2.0
    
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
    threshold, accuracy = find_optimal_separation_and_accuracy(final_endpoints, unique_classes, custom_threshold)
    
    # Draw separation line
    ax.axvline(x=threshold, color='yellow', linestyle='--', linewidth=1, alpha=0.8, zorder=5)
    
    # Display accuracy
    accuracy_text = f"Accuracy: {accuracy:.3f}"
    ax.text(0.02, 0.98, accuracy_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    # Draw shared U-axis
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8, zorder=1)

# Generate predictions and confusion matrices for both encodings
print("Generating predictions and confusion matrices...")

# GLC-L predictions and confusion matrix
glc_predictions, glc_cm, glc_threshold = generate_predictions_and_confusion_matrix(
    glc_l_encoding, X_norm_sorted, y, unique_classes
)

# GAC-L predictions and confusion matrix
gac_predictions, gac_cm, gac_threshold = generate_predictions_and_confusion_matrix(
    lambda x: gac_l_encoding(x, h=best_h), X_norm_sorted, y, unique_classes, best_threshold
)

# Print confusion matrices
print("\n" + "="*50)
print("CONFUSION MATRICES")
print("="*50)

print(f"\nGLC-L Confusion Matrix (Threshold: {glc_threshold:.3f}):")
print(glc_cm)
print(f"Accuracy: {np.sum(np.diag(glc_cm)) / np.sum(glc_cm):.3f}")

print(f"\nGAC-L Confusion Matrix (Threshold: {gac_threshold:.3f}):")
print(gac_cm)
print(f"Accuracy: {np.sum(np.diag(gac_cm)) / np.sum(gac_cm):.3f}")

# Print classification reports
print("\n" + "="*50)
print("CLASSIFICATION REPORTS")
print("="*50)

print(f"\nGLC-L Classification Report:")
print(classification_report(y, glc_predictions, target_names=[str(c) for c in unique_classes]))

print(f"\nGAC-L Classification Report:")
print(classification_report(y, gac_predictions, target_names=[str(c) for c in unique_classes]))

# Create 1x2 subplot layout (just the visualizations)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
titles = ['GLC-L (Length Encoding)', f'GAC-L (Angle Encoding, h={best_h:.2f})']

# Create legend handles for all classes
legend_handles = []
for i, class_label in enumerate(unique_classes):
    color_idx = class_to_index[class_label]
    handle = plt.Line2D([], [], color=colors[color_idx], linewidth=2, label=str(class_label))
    legend_handles.append(handle)

# Plot GLC-L visualization (left)
plot_with_shared_u_axis(axes[0], glc_l_encoding, X_norm_sorted, y, colors, class_to_index, unique_classes)
axes[0].set_title(titles[0] + "\n(Features sorted by LDA importance)")
axes[0].axis('equal')
axes[0].grid(True)
axes[0].set_facecolor('lightgrey')

# Plot GAC-L visualization (right)
plot_with_shared_u_axis(axes[1], lambda x: gac_l_encoding(x, h=best_h), X_norm_sorted, y, colors, class_to_index, unique_classes, best_threshold)
axes[1].set_title(titles[1] + "\n(Features sorted by LDA importance)")
axes[1].axis('equal')
axes[1].grid(True)
axes[1].set_facecolor('lightgrey')

# Add single legend to the figure
fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.suptitle(f"CSV Data Visualization in Two Encoding Schemes\n(Features sorted by LDA importance, {n_classes} classes)")
plt.tight_layout()
plt.show()