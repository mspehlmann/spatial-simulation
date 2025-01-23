import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt

#Define the spatial domain
n_points_X = 25
n_points_Y = 25

x = np.linspace(0, 10, n_points_X)
y = np.linspace(0, 10, n_points_Y)
X, Y = np.meshgrid(x, y)
locations = np.column_stack([X.ravel(), Y.ravel()])  #Shared locations
num_locations = len(locations)

#Kernel function
def gaussian_kernel(locations, range=1.0, variance=1.0):
    distances = cdist(locations, locations, metric="euclidean")
    return variance * np.exp(-0.5 * (distances / range) ** 2)

#Parameters for kernels
k = 3  #Number of categories
range_cat = 2
range_cont = 2
variance_cat = 2
variance_cont = 2

#Step 1: Simulate the categorical process
cov_cat = gaussian_kernel(locations=locations, range=range_cat, variance=variance_cat)
gp_samples_cat = np.random.multivariate_normal(mean=np.zeros(num_locations), cov=cov_cat, size=k)
gp_stacked = np.stack(gp_samples_cat, axis=-1)
categories = np.argmax(gp_stacked, axis=-1).reshape(n_points_X, n_points_Y)  # 2D array of categories

#Visualize the categorical process
plt.imshow(categories, extent=(0, 10, 0, 10), origin='lower', cmap="tab10")
plt.title("Categorical Process")
plt.colorbar(label="Category")
plt.show()

#Step 2: Compute distance to boundaries
#Compute a boundary mask
boundary_mask = np.zeros_like(categories, dtype=bool)

for i in range(categories.shape[0] - 1):
    for j in range(categories.shape[1] - 1):
        if categories[i, j] != categories[i + 1, j] or categories[i, j] != categories[i, j + 1]:
            boundary_mask[i, j] = True

#Compute distances to the nearest boundary
distances_to_boundary = distance_transform_edt(~boundary_mask)
distances_flat = distances_to_boundary.ravel()

#Visualize distances to boundaries
plt.imshow(distances_to_boundary, extent=(0, 10, 0, 10), origin='lower', cmap="viridis")
plt.title("Distance to Nearest Boundary")
plt.colorbar(label="Distance")
plt.show()

#Step 3: Simulate the continuous process with blending
#Define category-specific parameters
mean_vector = [0, 5, 10]  # Means for each category
variance_vector = [1, 2, 1.5]  # Variances for each category

#Allocate space for the continuous process
continuous_values = np.zeros(num_locations)

alpha = 1.0  #Controls blending sharpness
for i in range(num_locations):
    #Compute weights for all categories based on distance to the boundary
    weights = np.exp(-alpha * distances_flat[i])  #Exponential decay with distance
    weights /= np.sum(weights)  #Normalize weights to sum to 1

    #Compute blended mean and variance
    blended_mean = np.sum(weights * np.array(mean_vector))
    blended_variance = np.sum(weights * np.array(variance_vector))

    #Sample from the blended distribution
    continuous_values[i] = np.random.normal(blended_mean, np.sqrt(blended_variance))

#Visualize the continuous process
plt.scatter(locations[:, 0], locations[:, 1], c=continuous_values, cmap="viridis")
plt.title("Continuous Process with Blended Boundaries")
plt.colorbar(label="Continuous Value")
plt.show()