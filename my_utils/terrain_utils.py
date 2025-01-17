from math import sqrt
import torch 

import numpy as np
from numpy.random import choice
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics, UsdShade
from scipy import interpolate

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.prims import XFormPrim


def rooms_terrain(terrain, wall_height=1.0, wall_thickness=.2, passage_width=3.0):
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
    wall_thickness = int(wall_thickness/terrain.horizontal_scale)
    passage_width = int(passage_width/terrain.horizontal_scale)

    # Determine center points for room division and passage
    center_x = int(terrain_width // 2)
    center_y = int(terrain_length // 2)

    # Fill the entire height_field_raw with zeros (representing the floor)
    terrain.height_field_raw[:, :] = 0.0

    # Create walls around the perimeter of the terrain
    terrain.height_field_raw[:, 0:wall_thickness] = wall_height/terrain.vertical_scale  # Left wall
    terrain.height_field_raw[:, -wall_thickness:] = wall_height/terrain.vertical_scale  # Right wall
    terrain.height_field_raw[0:wall_thickness, :] = wall_height/terrain.vertical_scale  # Top wall
    terrain.height_field_raw[-wall_thickness:, :] = wall_height/terrain.vertical_scale  # Bottom wall

    # Create the internal wall separating the two rooms (leaving a passage in the middle)
    wall_start = center_y - wall_thickness // 2
    wall_end = center_y + wall_thickness // 2
    passage_start = center_x - passage_width // 2
    passage_end = center_x + passage_width // 2

    # Internal wall for the left and right rooms, except for the passage
    terrain.height_field_raw[wall_start:wall_end, :passage_start] = wall_height/terrain.vertical_scale  # Left room internal wall
    terrain.height_field_raw[wall_start:wall_end, passage_end:] = wall_height/terrain.vertical_scale  # Right room internal wall

    # Return updated terrain
    return terrain

def rooms_terrain2(terrain, wall_height=1.0, wall_thickness=.2, passage_width=0.0):
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
    wall_thickness = int(wall_thickness/terrain.horizontal_scale)
    passage_width = int(passage_width/terrain.horizontal_scale)

    # Determine center points for room division and passage
    center_x = int(terrain_width // 2)
    center_y = int(terrain_length // 2)

    # Fill the entire height_field_raw with zeros (representing the floor)
    terrain.height_field_raw[:, :] = 0.0

    # Create walls around the perimeter of the terrain
    terrain.height_field_raw[:, 0:wall_thickness] = wall_height/terrain.vertical_scale  # Left wall
    terrain.height_field_raw[:, -wall_thickness:] = wall_height/terrain.vertical_scale  # Right wall
    terrain.height_field_raw[0:wall_thickness, :] = wall_height/terrain.vertical_scale  # Top wall
    terrain.height_field_raw[-wall_thickness:, :] = wall_height/terrain.vertical_scale  # Bottom wall

    # Create the internal wall separating the two rooms (leaving a passage in the middle)
    wall_start = center_y - wall_thickness // 2
    wall_end = center_y + wall_thickness // 2
    passage_start = center_x - passage_width // 2
    passage_end = center_x + passage_width // 2

    # Internal wall for the left and right rooms, except for the passage
    terrain.height_field_raw[wall_start:wall_end, :passage_start] = wall_height/terrain.vertical_scale  # Left room internal wall
    terrain.height_field_raw[wall_start:wall_end, passage_end:] = wall_height/terrain.vertical_scale  # Right room internal wall

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


def mixed_pyramid_terrain_v1(terrain, num_steps, height_steps, slope, platform_width):
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
    # print("total_height", total_height)


    # Fill the platform area
    terrain.height_field_raw[platform_start_x:platform_stop_x, platform_start_y:platform_stop_y] = total_height

    # creta an array with the values from 0 to terrain.length
    y_slope_1 = np.arange(0, int(length_slope / terrain.horizontal_scale))
    coef1 = (((terrain.width * terrain.horizontal_scale) - platform_width)/2) / length_slope
    height = 0
    

    for y in y_slope_1: 
        y2 = (terrain.length - y) -1
        start_x = int(coef1 * y)
        stop_x = terrain.width - start_x
        height = int(y * slope)
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
        # print(start_y, stop_y)
        terrain.height_field_raw[x, start_y:stop_y] = height
        terrain.height_field_raw[x2, start_y:stop_y] = height

    height += int(height_steps / terrain.vertical_scale)

    for x in step_1:
        x2 = (terrain.width - x) -1
        start_y = int(coef2 * x)
        stop_y = terrain.length - start_y
        # print("height", height)
        terrain.height_field_raw[x, start_y:stop_y] = height
        terrain.height_field_raw[x2, start_y:stop_y] = height

    return terrain


def mixed_pyramid_terrain_v2(terrain, num_steps, height_steps, slope, platform_width):
    # height_steps = int(height_steps / terrain.vertical_scale)
    # platform_width = int(platform_width / terrain.horizontal_scale)

    # Calculate the total height of the pyramid
    total_height = num_steps * height_steps

    # Calculate the length of the slope sides
    length_slope = total_height / slope

    # Calculate the dynamic platform length
    platform_length = terrain.length - 2 * int(length_slope / terrain.horizontal_scale)

    # Ensure the platform fits within the terrain
    if platform_length < 0 or platform_width < 0:
        raise ValueError("The calculated platform dimensions are invalid. Adjust the input parameters.")

    # Calculate the start and stop coordinates for the platform
    platform_start_x = (terrain.width - int(platform_width / terrain.horizontal_scale)) // 2
    platform_stop_x = (platform_start_x + int(platform_width / terrain.horizontal_scale))
    platform_start_y = (terrain.length - platform_length) // 2
    platform_stop_y = (platform_start_y + platform_length)
    total_height = int(total_height / terrain.vertical_scale)
    # print("total_height", total_height)


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
        start_x = int((((terrain.width * terrain.horizontal_scale) - platform_width)/2)/terrain.horizontal_scale)
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
        start_y = int((((terrain.length * terrain.horizontal_scale) - platform_width)/2)/terrain.horizontal_scale)
        stop_y = terrain.length - start_y
        # print(start_y, stop_y)
        terrain.height_field_raw[x, start_y:stop_y] = height
        terrain.height_field_raw[x2, start_y:stop_y] = height

    height += int(height_steps / terrain.vertical_scale)

    for x in step_1:
        x2 = (terrain.width - x) -1
        start_y = int((((terrain.length * terrain.horizontal_scale) - platform_width)/2)/terrain.horizontal_scale)
        stop_y = terrain.length - start_y
        # print("height", height)
        terrain.height_field_raw[x, start_y:stop_y] = height
        terrain.height_field_raw[x2, start_y:stop_y] = height

    return terrain


def mixed_pyramid_terrain_v3(terrain, slope, platform_size):

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)

    height_temp = 1

    for i in x:
        for j in y:
            if i > (center_x + 1) and j < (center_y - 1):
                terrain.height_field_raw[i, j] = height_temp
            elif i < (center_x - 1) and j > (center_y + 1):
                terrain.height_field_raw[i, j] = height_temp

    max_value = np.max(terrain.height_field_raw)
    # print("Maximum value in terrain.height_field_raw:", max_value)

    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y

    xx1 = xx.reshape(terrain.width, 1)
    yy1 = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw[:(center_x + 1), :(center_y)] += (max_height * xx1 * yy1)[:(center_x + 1), :(center_y)].astype(terrain.height_field_raw.dtype)
    terrain.height_field_raw[(center_x):, (center_y):] += (max_height * xx1 * yy1)[(center_x):, (center_y):].astype(terrain.height_field_raw.dtype)

      
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain

def custom_sloped_terrain(terrain, height_steps=0.5, slope=0.08, platform_size=1.0):
    num_steps=2
    # generate a very simple terrain that has a slope universal over y going from 0 to max_height
    max_height = num_steps * height_steps
    horizontal_slope_distance = int((max_height/terrain.vertical_scale) / (slope * (terrain.horizontal_scale / terrain.vertical_scale)))

    x1 =  int((terrain.width - int(platform_size/terrain.horizontal_scale)) / 2 - horizontal_slope_distance)      # flat before slope   
    x2 = np.arange(x1, x1 + int(horizontal_slope_distance)) # slope
    x3 = np.arange(x1 + int(horizontal_slope_distance), x1 + int(horizontal_slope_distance) + int(platform_size/terrain.horizontal_scale))
    x5 = int(terrain.width - ((terrain.width - (x1 + int(horizontal_slope_distance) + int(platform_size/terrain.horizontal_scale)))/2))
    x4 = np.arange(x1 + int(horizontal_slope_distance) + int(platform_size/terrain.horizontal_scale), x5) # flat after slope


    y = np.arange(0, int(terrain.length))

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x2, y, sparse=True)
    xx = xx.reshape(x2.size, 1)
    x2 = x1 + int(horizontal_slope_distance)
    terrain.height_field_raw[x1:x2, np.arange(terrain.length)] += (slope * (terrain.horizontal_scale / terrain.vertical_scale) * (xx-x1)).astype(
        terrain.height_field_raw.dtype
    )

    # Calculate the expression
    height_update = (num_steps * height_steps / terrain.vertical_scale)

    # Ensure the shape matches the section of terrain.height_field_raw being updated
    height_update = np.full((x3.size, terrain.length), height_update)
    x3 = (x1 + int(horizontal_slope_distance) + int(platform_size/terrain.horizontal_scale))
    terrain.height_field_raw[x2:x3, np.arange(terrain.length)] += height_update.astype(
        terrain.height_field_raw.dtype
    )

    # Calculate the expression
    height_update = height_steps / terrain.vertical_scale

    # Ensure the shape matches the section of terrain.height_field_raw being updated
    height_update = np.full((x4.size, terrain.length), height_update)
    x4 = int(terrain.width - ((terrain.width - (x1 + int(horizontal_slope_distance) + int(platform_size/terrain.horizontal_scale)))/2))
    terrain.height_field_raw[x3:x4, np.arange(terrain.length)] += height_update.astype(
        terrain.height_field_raw.dtype
    )

    return terrain

def custom_mixed_terrain(terrain, num_steps, height_steps, slope, platform_width):
    # generate a very simple terrain that has a slope universal over y going from 0 to max_height
    max_height = num_steps * height_steps
    horizontal_distance = int((max_height / slope) / terrain.horizontal_scale)

    # print(terrain.height_field_raw.shape)
    # print(horizontal_distance)

        
    x = np.arange(0, horizontal_distance)
    y = np.arange(0, int(terrain.length))

    for i in x:
        for j in y:
            height_value = int(slope * i )
            terrain.height_field_raw[i, j] = height_value

    x1 = np.arange(horizontal_distance, (horizontal_distance + int(platform_width/terrain.horizontal_scale)))
    height_value = int(max_height / terrain.vertical_scale)

    # print(x1)
    # print(y)
    # print(terrain.height_field_raw.shape)

    for i in x1:
        for j in y:
            terrain.height_field_raw[i, j] = height_value

    step_width = (((terrain.width - (horizontal_distance + (platform_width/ terrain.horizontal_scale))) / num_steps))


    for z in range(num_steps):
        x2 = np.arange(int(horizontal_distance + (platform_width/ terrain.horizontal_scale) + z * step_width), int(horizontal_distance + (platform_width/ terrain.horizontal_scale) + (z+1)*step_width))
        for i in x2:
            for j in y:
                height_step = int((max_height/terrain.horizontal_scale) / (z+1))
                terrain.height_field_raw[i, j] = height_step

    return terrain


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles


def add_terrain_to_stage(stage, vertices, triangles, position=None, orientation=None):
    num_faces = triangles.shape[0]
    terrain_mesh = stage.DefinePrim("/World/terrain", "Mesh")
    terrain_mesh.GetAttribute("points").Set(vertices)
    terrain_mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten())
    terrain_mesh.GetAttribute("faceVertexCounts").Set(np.asarray([3] * num_faces))

    terrain = XFormPrim(prim_path="/World/terrain", name="terrain", position=position, orientation=orientation)

    UsdPhysics.CollisionAPI.Apply(terrain.prim)
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(terrain.prim)
    collision_api.CreateApproximationAttr().Set("meshSimplification")
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
    physx_collision_api.GetContactOffsetAttr().Set(0.02)
    physx_collision_api.GetRestOffsetAttr().Set(0.00)


    # material_path = "/World/Looks/TerrainMaterial"
    # material_prim = stage.DefinePrim(material_path, "Material")
    # material_prim.GetReferences().AddReference(
    #     "omniverse://localhost/NVIDIA/Materials/2023_1/vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl"
    # )
    # UsdShade.MaterialBindingAPI(terrain.prim).Bind(UsdShade.Material(material_prim))


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c