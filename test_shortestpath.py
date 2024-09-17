import numpy as np
from PIL import Image, ImageDraw
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import subprocess as s_proc
from alive_progress import alive_bar
import math

s_proc.run(['python', 'heightmap_gen.py'])

colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)] #green lowest weight, red highest weight
cmap = LinearSegmentedColormap.from_list("heightmap_cmap", colors)
norm = plt.Normalize(vmin=0, vmax=1)
color_array = np.array([cmap(norm(i)) for i in np.linspace(0, 1, 256)])

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    image_array = np.array(img)
    normalized_image_array = image_array / 255.0

    return np.array(normalized_image_array)

def color_to_weight(color):

    rgba_color = np.array(to_rgba(color)) 

    differences = np.linalg.norm(np.array(color_array) - rgba_color, axis=1)
    closest_index = np.argmin(differences)
    
    float = closest_index / (len(color_array) - 1)*10.

    if color[0] > 0.9 and color[1] > 0.9 and color[2] > 0.9:
        float = math.inf
    return float

def create_grid(image_array, grid_size):
    h, w, _ = image_array.shape
    nodes = np.zeros((h // grid_size, w // grid_size))

    with alive_bar((h // grid_size) * (w // grid_size), title='Creating Grid: ', length=20) as bar: 
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                color = image_array[i, j]
                nodes[i // grid_size, j // grid_size] = color_to_weight(color)
                bar()
    return nodes

def get_neighbors(node, grid):
    neighbors = []
    x, y = node
    if x > 0: neighbors.append((x - 1, y))
    if x < grid.shape[0] - 1: neighbors.append((x + 1, y))
    if y > 0: neighbors.append((x, y - 1))
    if y < grid.shape[1] - 1: neighbors.append((x, y + 1))
    return neighbors

def dijkstra(grid, start, goal):
    h, w = grid.shape
    distance = np.full((h, w), float('inf'))
    distance[start] = 0 
    priority_queue = [(0, start)]  
    came_from = {} 
    total_work = 0

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)
        
        if current_node == goal:
            break

        for neighbor in get_neighbors(current_node, grid):
            new_cost = current_dist + grid[neighbor] 
            total_work += new_cost
            if new_cost < distance[neighbor]:
                distance[neighbor] = new_cost
                heapq.heappush(priority_queue, (new_cost, neighbor))
                came_from[neighbor] = current_node

    if goal not in came_from:
        return []

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    return path, total_work


def draw_path(image_array, path, grid_size):
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    with alive_bar(len(path) - 1, title='Drawing Path: ', length=20) as bar: 
        for i in range(1, len(path)):
            start = (path[i-1][1] * grid_size + grid_size // 2, path[i-1][0] * grid_size + grid_size // 2)
            end = (path[i][1] * grid_size + grid_size // 2, path[i][0] * grid_size + grid_size // 2)
            draw.line([start, end], fill=(0, 0, 0), width=4)
            bar()
    
    return np.array(img)

def on_hover(event, ax, image_array, grid_size, ax_text, rows, cols):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        row, col = y // grid_size, x // grid_size

        if 0 <= row < rows and 0 <= col < cols:
            ax_text.set_text(f'Grid: ({row}, {col})')
        else:
            ax_text.set_text('')
    plt.draw()

def main(image_path, grid_size, start, goal):
    image_array = load_image(image_path)
    grid = create_grid(image_array, grid_size)
    path = dijkstra(grid, start, goal)[0]
    # print("Total Work: " + str(round(dijkstra(grid, start, goal)[1])))
    result_image = draw_path(image_array, path, grid_size)

    fig, ax = plt.subplots()
    ax.imshow(result_image)
    plt.axis('on')

    rows, cols = result_image.shape[0] // grid_size, result_image.shape[1] // grid_size
    ax_text = ax.text(1.30, 0.05, '', color='white', ha='right', va='bottom', 
                      transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

    fig.canvas.mpl_connect('motion_notify_event', lambda event: on_hover(event, ax, result_image, grid_size, ax_text, rows, cols))

    plt.show()

if __name__ == "__main__":
    image_path = 'heightmap.jpg'
    grid_size = 1

    i_size, _, _ = load_image(image_path).shape
    start = (0, 0)
    goal = ((i_size / grid_size) - 1, (i_size / grid_size) - 1)



    main(image_path, grid_size, start, goal)
