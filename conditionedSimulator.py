import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
import pandas as pd
import seaborn as sns

class CategoricalProcess:
    def __init__(self, grid_x, grid_y, n_cateogories, kernel_range, kernel_variance, seed=None):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.n_categories = n_cateogories
        self.kernel_range = kernel_range
        self.kernel_variance = kernel_variance
        self.locations = self._create_grid()
        self.boundary_distance = None
        self.categorical_process = None
        if seed is not None:
            np.random.seed(seed)
        
    def _create_grid(self):
        x = np.linspace(0, 10, self.grid_x)
        y = np.linspace(0, 10, self.grid_y)
        X, Y = np.meshgrid(x, y)
        self.X, self.Y = X, Y
        return np.column_stack([X.ravel(), Y.ravel()])
        
    def see_spatial_domain(self):
        if self.locations is None:
            self._create_grid()
        plt.scatter(self.locations[:, 0], self.locations[:, 1], c="#033762")
        plt.title("Spatial domain")
        plt.show()

    def _guassian_kernel(self):
        dists = cdist(self.locations, self.locations, metric="euclidean")
        return self.kernel_variance * np.exp(-0.5 * (dists / self.kernel_range)**2)
    
    def simulate(self):
        cov = self._guassian_kernel()
        gp_samples = np.random.multivariate_normal(mean = np.zeros(len(self.locations)), cov=cov, size=self.n_categories)
        stacked = np.stack(gp_samples, axis=-1)
        self.categorical_process = np.argmax(stacked, axis=-1).reshape(self.grid_x, self.grid_y)
        self._compute_boundary_distance()
        return self.categorical_process
    
    def _compute_boundary_distance(self):
        boundary_mask = np.zeros_like(self.categorical_process, dtype=bool)
        for i in range(self.categorical_process.shape[0] -1):
            for j in range(self.categorical_process.shape[1] - 1):
                if self.categorical_process[i, j] != self.categorical_process[i + 1, j] or self.categorical_process[i, j] != self.categorical_process[i, j + 1]:
                    boundary_mask[i, j] = True
        self.boundary_distance = distance_transform_edt(~boundary_mask)

    def see_categorical_process(self):
        fig, ax_cat = plt.subplots(figsize=(5, 5))
        cmap = plt.get_cmap('tab10', self.n_categories)
        ax_cat.imshow(self.categorical_process, cmap=cmap, interpolation='nearest')
        ax_cat.set_title("Categorical Process")
        ax_cat.axis('off')
        plt.colorbar(ax_cat.imshow(self.categorical_process, cmap=cmap, interpolation='nearest'), ax=ax_cat, ticks=np.arange(self.n_categories), label='Categories')
        plt.show()

class continuousProcess:
    def __init__(self, categorical_process: CategoricalProcess, mean_vector, range_vector, variance_vector, alpha = 1.0, blending_threshold=0.5):
        self.categorical_process = categorical_process
        self.mean_vector = mean_vector
        self.range_vector = range_vector
        self.variance_vector = variance_vector
        self.process_field = None
        self.alpha = alpha
        self.blending_threshold = blending_threshold
    
    def _kernel(self, range_val, var_val):
        locs = self.categorical_process.locations
        dists = cdist(locs, locs, metric="euclidean")
        return var_val * np.exp(-0.5 * (dists / range_val) ** 2)
    
    @staticmethod
    def _add_noise(value=0.001, std_dev=0.1, absolute=True):
        noise = np.random.normal(0, std_dev)
        if absolute:
            return np.abs(value + noise)
        else:
            return value + noise
    
    def simulate(self):
        categories = self.categorical_process.categorical_process.ravel()
        dists_to_boundary = self.categorical_process.boundary_distance.ravel()
        num_locations = len(categories)

        values = np.zeros(num_locations)

        for i in range(num_locations):
            cat = categories[i]
            base_mean = self.mean_vector[cat]
            base_var = self.variance_vector[cat]
            dist = dists_to_boundary[i]

            x_idx = i // self.categorical_process.grid_y
            y_idx = i % self.categorical_process.grid_y

            # Coordinates of the current point
            current_point = np.array([x_idx, y_idx])
            current_cat = categories[i]

            # Initialize search radius
            max_radius = 5  # Increase if necessary
            neighbor_cat = current_cat  # fallback in case no neighbor is found

            for r in range(1, max_radius + 1):
                # Get neighborhood window
                x_min = max(0, x_idx - r)
                x_max = min(self.categorical_process.grid_x, x_idx + r + 1)
                y_min = max(0, y_idx - r)
                y_max = min(self.categorical_process.grid_y, y_idx + r + 1)

                subgrid = self.categorical_process.categorical_process[x_min:x_max, y_min:y_max]
                for xx in range(x_min, x_max):
                    for yy in range(y_min, y_max):
                        if self.categorical_process.categorical_process[xx, yy] != current_cat:
                            neighbor_cat = self.categorical_process.categorical_process[xx, yy]
                            break
                    else:
                        continue
                    break

                if neighbor_cat != current_cat:
                    break  

            if dist < self.blending_threshold:
                neighbor_cats = [c for c in range(len(self.mean_vector)) if c != cat]
                #neighbor = neighbor_cats[0]
                
                neighbor_mean = self.mean_vector[neighbor_cat]
                neighbor_var = self.variance_vector[neighbor_cat]

                weight = np.exp(-self.alpha * (dist + self._add_noise()))

                blended_mean = (1 - weight) * base_mean + weight * neighbor_mean
                blended_var = (1 - weight) * base_var + weight * neighbor_var

                blended_var = base_var + weight * blended_var
                values[i] = np.random.normal(loc=blended_mean, scale=np.sqrt(blended_var))

                print("The weight is {}".format(weight))
                print("The base mean is {} and the blended mean is {}.".format
                      (base_mean, blended_mean))
                print("The base variance is {} and the blended variance is {}".format(base_var, blended_var))
            else:
                values[i] = np.random.normal(loc=base_mean, scale=np.sqrt(base_var))

        self.process_field = values.reshape(self.categorical_process.grid_x, self.categorical_process.grid_y)
        return self.process_field
    
    def see_dist_by_category(self):
        if self.process_field is None:
            raise ValueError("Process field not simulated yet. Call simulate() first.")
        df = pd.DataFrame({
            "Continuous Value": self.process_field.ravel(),
            "Category": self.categorical_process.categorical_process.ravel()
        })

        plt.figure(figsize=(7, 4))
        for cat in range(self.categorical_process.n_categories):
            subset = df[df["Category"] == cat]
            if not subset.empty:
                sns.kdeplot(subset["Continuous Value"], label=f"Category {cat}", fill=True)
                plt.axvline(self.mean_vector[cat], linestyle="dashed", color="black", alpha=0.7)

        plt.xlabel("Continuous Value")
        plt.ylabel("Density")
        plt.title("KDE of Continuous Values per Category")
        plt.legend()
        plt.show()
    
    def see_conditioned_continuous_process(self):
        if self.process_field is None:
            raise ValueError("Process field not simulated yet. Call simulate() first.")
        
        X = self.categorical_process.X
        Y = self.categorical_process.Y
        cats = self.categorical_process.categorical_process

        plt.figure(figsize=(6, 5))

        plt.imshow(self.process_field, extent=(0, 10, 0, 10), origin="lower",
                cmap="seismic", interpolation="nearest")
        
        plt.colorbar(label="Continuous Value")

        plt.contour(X, Y, cats, levels=np.arange(0.5, self.categorical_process.n_categories), 
                    colors="black", linewidths=1.5)

        plt.title("Continuous Process with Category Boundaries")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

firstCatProcess = CategoricalProcess(grid_x=25, grid_y=25, n_cateogories=3, kernel_range=1.5, kernel_variance=0.5, seed=42)
#firstCatProcess.see_spatial_domain()
firstCatProcess.simulate()
#firstCatProcess.see_categorical_process()

firstContProcess = continuousProcess(categorical_process=firstCatProcess, mean_vector=[0, 10, 20], range_vector=[1.5, 1.5, 2.5], 
                                     variance_vector=[1.25, 1.25, 1.25], 
                                     alpha=1.5, blending_threshold=1)

firstContProcess.simulate()
firstContProcess.see_dist_by_category()
firstContProcess.see_conditioned_continuous_process()
