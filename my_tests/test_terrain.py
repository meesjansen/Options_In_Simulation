import math

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import choice



def rooms_terrain(terrain, wall_height=400, wall_thickness=5, passage_width=20):
    """
    Generate a terrain with two fully enclosed rooms connected by a passage, based on the grid setup provided.

    Parameters:
        terrain (SubTerrain): the terrain object
        wall_height (int): height of the walls enclosing the rooms (default: 200 units)
        wall_thickness (int): thickness of the walls (default: 1 unit)
        passage_width (int): width of the passage connecting the rooms (default: 2 units)
    Returns:
        terrain (SubTerrain): updated terrain
    """
    # Get the terrain dimensions
    terrain_width = terrain.width
    terrain_length = terrain.length
    wall_thickness = int(wall_thickness)
    passage_width = int(passage_width)

    # Determine center points for room division and passage
    center_x = int(terrain_width // 2)
    center_y = int(terrain_length // 2)

    # Fill the entire height_field_raw with zeros (representing the floor)
    terrain.height_field_raw[:, :] = 0

    # Create walls around the perimeter of the terrain
    terrain.height_field_raw[:, 0:wall_thickness] = wall_height  # Left wall
    terrain.height_field_raw[:, -wall_thickness:] = wall_height  # Right wall
    terrain.height_field_raw[0:wall_thickness, :] = wall_height  # Top wall
    terrain.height_field_raw[-wall_thickness:, :] = wall_height  # Bottom wall

    # Create the internal wall separating the two rooms (leaving a passage in the middle)
    wall_start = center_y - wall_thickness // 2
    wall_end = center_y + wall_thickness // 2
    passage_start = center_x - passage_width // 2
    passage_end = center_x + passage_width // 2

    # Internal wall for the left and right rooms, except for the passage
    terrain.height_field_raw[wall_start:wall_end, :passage_start] = wall_height  # Left room internal wall
    terrain.height_field_raw[wall_start:wall_end, passage_end:] = wall_height  # Right room internal wall

    # Return updated terrain
    return terrain



def stairs_terrain(terrain, step_width, step_height):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width : (i + 1) * step_width, :] += height
        height += step_height
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.0):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
    return terrain


def custom_pyramid_terrain(terrain, num_steps, height_steps, slope, platform_width):
    # height_steps = int(height_steps / terrain.vertical_scale)
    # platform_width = int(platform_width / terrain.horizontal_scale)

    # Calculate the total height of the pyramid
    total_height = num_steps * height_steps

    # Calculate the length of the slope sides
    length_slope = total_height / slope

    # Calculate the dynamic platform length
    platform_length = int(terrain.length / terrain.horizontal_scale) - 2 * int(length_slope / terrain.horizontal_scale)

    # Ensure the platform fits within the terrain
    if platform_length < 0 or platform_width < 0:
        raise ValueError("The calculated platform dimensions are invalid. Adjust the input parameters.")

    # Calculate the start and stop coordinates for the platform
    platform_start_x = (terrain.width - int(platform_width / terrain.horizontal_scale)) // 2
    platform_stop_x = (platform_start_x + int(platform_width / terrain.horizontal_scale))
    platform_start_y = (terrain.length - platform_length) // 2
    platform_stop_y = (platform_start_y + platform_length)
    total_height = int(total_height / terrain.vertical_scale)
    print("total_height", total_height)


    # Fill the platform area
    terrain.height_field_raw[platform_start_x:platform_stop_x, platform_start_y:platform_stop_y] = total_height

    # creta an array with the values from 0 to terrain.length
    y_slope_1 = np.arange(0, int(length_slope / terrain.horizontal_scale))
    coef1 = (((terrain.width * terrain.horizontal_scale) - platform_width)/2) / length_slope
    height = 0
    

    for y in y_slope_1: 
        # print("y", y)   
        y2 = (terrain.length - y) -1
        # print("y2", y2)
        start_x = int(coef1 * y)
        stop_x = terrain.width - start_x
        # print(start_x, stop_x)
        height = int(y * slope)
        # print("height slope:", height)
        terrain.height_field_raw[start_x:stop_x, y] = height
        terrain.height_field_raw[start_x:stop_x, y2] = height
 
    x_slope_1 = np.arange(0, int((terrain.width - (platform_width / terrain.horizontal_scale))/2))
    step_width = int(len(x_slope_1) / num_steps)
    step_0 = x_slope_1[::step_width]
    step_1 = x_slope_1[step_width:]
    coef2 = 1 / coef1
    height = 0

    for x in step_0:
        x2 = (terrain.width - x) -1
        start_y = int(coef2 * x)
        stop_y = terrain.length - start_y
        print(start_y, stop_y)
        terrain.height_field_raw[x, start_y:stop_y] = height
        terrain.height_field_raw[x2, start_y:stop_y] = height

    height += int(height_steps / terrain.vertical_scale)

    for x in step_1:
        x2 = (terrain.width - x) -1
        start_y = int(coef2 * x)
        stop_y = terrain.length - start_y
        print("height", height)
        terrain.height_field_raw[x, start_y:stop_y] = height
        terrain.height_field_raw[x2, start_y:stop_y] = height

    
    
    return terrain


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)



class Terrain:
    def __init__(self, num_robots) -> None:
        self.horizontal_scale = 0.005
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 1
        self.env_length = 4.5
        self.env_width = self.env_length

        self.env_rows = int(math.sqrt(num_robots))
        self.env_cols = int(math.sqrt(num_robots))
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # rooms, stairs, sloped, mixed
        terrain_type = task_cfg["env"]["TerrainType"]
        self.cr_env(terrain_type)
        
        self.heightsamples = self.height_field_raw
        
    def cr_env(self, terrain_type):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.width_per_env_pixels
            end_x = self.border + (i + 1) * self.width_per_env_pixels
            start_y = self.border + j * self.length_per_env_pixels
            end_y = self.border + (j + 1) * self.length_per_env_pixels

            terrain = SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            if terrain_type == "rooms":
                rooms_terrain(terrain, wall_height=50, wall_thickness=5, passage_width=20)
            elif terrain_type == "stairs":
                pyramid_stairs_terrain(terrain, step_width=1.0, step_height=0.08, platform_size=1.5)
            elif terrain_type == "sloped":
                pyramid_sloped_terrain(terrain, slope=1, platform_size=1.0)
            elif terrain_type == "mixed":
                custom_pyramid_terrain(terrain, num_steps=2, height_steps=0.08, slope=0.1, platform_width=1.5)
            else:
                raise ValueError(f"Unknown TerrainType: {terrain_type}")


            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2.0 - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2.0 + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


class TerrainPlotter:
    def __init__(self, terrain_generator):
        self.terrain_generator = terrain_generator
        self.height_field_raw = self.terrain_generator.heightsamples
        self.vertical_scale = self.terrain_generator.vertical_scale

    def plot_terrain(self):
        height_field_scaled = self.height_field_raw * self.vertical_scale
        plt.imshow(height_field_scaled, cmap='coolwarm')
        plt.colorbar(label='Height')
        plt.title('Terrain Height Field')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()


if __name__ == "__main__":
    task_cfg = {
        "env": {
            "TerrainType": "mixed", # rooms, stairs, sloped, mixed
        }
    }
    
    terrain_generator = Terrain(1)
    plotter = TerrainPlotter(terrain_generator)
    plotter.plot_terrain()