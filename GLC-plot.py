import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load and normalize
iris = load_iris()
X = iris.data  # shape (150, 4)
y = iris.target
feature_names = iris.feature_names

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
colors = ['tab:blue', 'tab:orange', 'tab:green']

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
    for class_id in np.unique(y):
        X_class = X_norm_sorted[y == class_id]
        for row in X_class[:10]:  # Plot first 10 samples per class for clarity
            path = encoding_func(row)
            ax.plot(path[:, 0], path[:, 1], alpha=0.6, color=colors[class_id])
    ax.set_title(titles[idx] + "\n(Features sorted by LDA importance)")
    ax.axis('equal')
    ax.grid(True)

plt.suptitle("Fisher Iris Visualization in Two Encoding Schemes\n(Features sorted by LDA importance)")
plt.tight_layout()
plt.show()