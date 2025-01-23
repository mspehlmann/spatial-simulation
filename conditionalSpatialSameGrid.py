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


cov_cat = gaussian_kernel(locations=locations, range=range_cat, variance=variance_cat)
gp_samples_cat = np.random.multivariate_normal(mean=np.zeros(num_locations), cov=cov_cat, size=k)
gp_stacked = np.stack(gp_samples_cat, axis=-1)
categories = np.argmax(gp_stacked, axis=-1).reshape(n_points_X, n_points_Y)  #2D array of categories

#Visualize the categorical process
plt.imshow(categories, extent=(0, 10, 0, 10), origin='lower', cmap="tab10")
plt.title("Categorical Process")
plt.colorbar(label="Category")
plt.show()


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


#Define category-specific parameters
mean_vector = [0, 50, 10]  #Means for each category
variance_vector = [1, 2, 1.5]  #Variances for each category

continuous_values = np.zeros(num_locations)

alpha = 1.0  # Controls blending sharpness
blending_threshold = 2  # Distance threshold for blending

#Simulate a spatially correlated field for each category
cov_cont = gaussian_kernel(locations, range=range_cont, variance=variance_cont)
spatial_fields = np.random.multivariate_normal(mean=np.zeros(num_locations), cov=cov_cont, size=k)

for i in range(num_locations):
    #Get the category for the current location
    assigned_category = categories.ravel()[i]

    #Compute the base mean and variance based on the assigned category
    base_mean = mean_vector[assigned_category]
    base_variance = variance_vector[assigned_category]

    #Compute distance to the nearest boundary
    dist_to_boundary = distances_to_boundary.ravel()[i]

    #Blending near the boundary
    if dist_to_boundary < blending_threshold:  #Blending region
        #Compute weights for blending with other categories
        blended_mean = 0
        blended_variance = 0
        total_weight = 0
        for cat in range(k):  #Blend contributions from all categories
            weight = np.exp(-alpha * dist_to_boundary if cat != assigned_category else 0)
            blended_mean += weight * mean_vector[cat]
            blended_variance += weight * variance_vector[cat]
            total_weight += weight
        blended_mean /= total_weight
        blended_variance /= total_weight

        #Sample from the blended distribution
        continuous_values[i] = np.random.normal(blended_mean, np.sqrt(blended_variance))
    else:  #Outside blending threshold: use spatially correlated field
        continuous_values[i] = spatial_fields[assigned_category, i] + base_mean

#Visualize the continuous process
plt.scatter(locations[:, 0], locations[:, 1], c=continuous_values, cmap="viridis", s=50, edgecolor="k")
plt.title("Continuous Process with Spatial Dependence and Blended Boundaries")
plt.colorbar(label="Continuous Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()