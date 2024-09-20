import math

import numpy as np
import torch
from my_utils.terrain_utils import *

class Terrain:
    def __init__(self, num_robots) -> None:
        self.horizontal_scale = 0.3
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 1
        self.env_length = 4
        self.env_width = 2
        # self.proportions = [np.sum(cfg["terrainProportions"][: i + 1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = 1
        self.env_cols = 1
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.cr_rooms_env()

        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
                    self.height_field_raw, self.horizontal_scale, self.vertical_scale, slope_threshold=0.5)


    def cr_rooms_env(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            rooms_terrain(terrain, wall_height=100, wall_thickness=5, passage_width=20)

            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2.0 - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2.0 + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

            