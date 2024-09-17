import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from alive_progress import alive_bar

def diamond_square(size, roughness):

    height_map = np.zeros((size, size))

    height_map[0, 0] = np.random.rand()
    height_map[0, size - 1] = np.random.rand()
    height_map[size - 1, 0] = np.random.rand()
    height_map[size - 1, size - 1] = np.random.rand()

    step_size = size - 1
    scale = roughness

    with alive_bar(1050621, title='Creating Map: ', length=20) as bar:
        while step_size > 1:
            half_step = step_size // 2

            for x in range(0, size - 1, step_size):
                for y in range(0, size - 1, step_size):
                    avg = (height_map[x, y] +
                       height_map[x + step_size, y] +
                       height_map[x, y + step_size] +
                       height_map[x + step_size, y + step_size]) / 4.0
                    height_map[x + half_step, y + half_step] = avg + (np.random.rand() - 0.5) * scale
                    bar()
                

            for x in range(0, size, half_step):
                for y in range((x + half_step) % step_size, size, step_size):
                    avg = (height_map[(x - half_step) % (size - 1), y] +
                       height_map[(x + half_step) % (size - 1), y] +
                       height_map[x, (y - half_step) % (size - 1)] +
                       height_map[x, (y + half_step) % (size - 1)]) / 4.0
                    height_map[x, y] = avg + (np.random.rand() - 0.5) * scale
                    bar()

            step_size //= 2
            scale /= 1.4

    min_val, max_val = height_map.min(), height_map.max()
    height_map = (height_map - min_val) / (max_val - min_val)
    return height_map

size = 1025 
roughness = 1.0
height_map = diamond_square(size, roughness)

colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)] 
cmap = LinearSegmentedColormap.from_list("heightmap_cmap", colors)

plt.figure(figsize=(1, 1)) 
plt.imshow(height_map, cmap=cmap)
plt.axis('off') 

plt.savefig("heightmap.jpg", bbox_inches='tight', pad_inches=0, dpi=5000)

# plt.show()
