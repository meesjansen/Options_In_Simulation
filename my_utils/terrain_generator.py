import math

import numpy as np
import torch
from my_utils.terrain_utils import *


class Terrain:
    def __init__(self, num_robots, terrain_type) -> None:
        self.horizontal_scale = 0.005
        self.vertical_scale = 0.0005
        self.border_size = 20
        self.num_per_env = 1
        self.env_length = 8.0
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
        self.cr_env(terrain_type)
        
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
                    self.height_field_raw, self.horizontal_scale, self.vertical_scale, slope_threshold=None)


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
            elif terrain_type == "mixed_v1":
                mixed_pyramid_terrain_v1(terrain, num_steps=2, height_steps=0.08, slope=0.1, platform_width=1.5)
            elif terrain_type == "mixed_v2":
                mixed_pyramid_terrain_v2(terrain, num_steps=2, height_steps=0.08, slope=0.06, platform_width=1.5)
            elif terrain_type == "mixed_v3":
                mixed_pyramid_terrain_v3(terrain, slope=0.5, platform_size=1.0)
            elif terrain_type == "custom":
                custom_sloped_terrain(terrain, height_steps=0.08, slope=.1, platform_size=1.0)
            elif terrain_type == "custom_mixed":
                custom_mixed_terrain(terrain, num_steps=2, height_steps=0.08, slope=0.06, platform_width=1.5)
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

    # def cr_steps_env(self):
    #     for k in range(self.num_maps):
    #         # Env coordinates in the world
    #         (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

    #         # Heightfield coordinate system from now on
    #         start_x = self.border + i * self.width_per_env_pixels
    #         end_x = self.border + (i + 1) * self.width_per_env_pixels
    #         start_y = self.border + j * self.length_per_env_pixels
    #         end_y = self.border + (j + 1) * self.length_per_env_pixels

    #         terrain = SubTerrain(
    #             "terrain",
    #             width=self.width_per_env_pixels,
    #             length=self.length_per_env_pixels,
    #             vertical_scale=self.vertical_scale,
    #             horizontal_scale=self.horizontal_scale,
    #         )

    #         pyramid_stairs_terrain(terrain, step_width=1.0, step_height=0.08, platform_size=1.5)

    #         self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    #         env_origin_x = (i + 0.5) * self.env_length
    #         env_origin_y = (j + 0.5) * self.env_width
    #         x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
    #         x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
    #         y1 = int((self.env_width / 2.0 - 1) / self.horizontal_scale)
    #         y2 = int((self.env_width / 2.0 + 1) / self.horizontal_scale)
    #         env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
    #         self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    # def cr_mixed_terrain_env(self):
    #     """
    #     Generate an environment with the mixed pyramid terrain where two sides have steps, and the other two are sloped.
    #     """
    #     for k in range(self.num_maps):
    #         # Env coordinates in the world
    #         (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

    #         # Heightfield coordinate system
    #         start_x = self.border + i * self.width_per_env_pixels
    #         end_x = self.border + (i + 1) * self.width_per_env_pixels
    #         start_y = self.border + j * self.length_per_env_pixels
    #         end_y = self.border + (j + 1) * self.length_per_env_pixels

    #         terrain = SubTerrain(
    #             "terrain",
    #             width=self.width_per_env_pixels,
    #             length=self.length_per_env_pixels,
    #             vertical_scale=self.vertical_scale,
    #             horizontal_scale=self.horizontal_scale,
    #         )

    #         # Generate mixed pyramid terrain
    #         mixed_pyramid_terrain(terrain, step_width=1.0, step_height=0.08, slope=1, platform_size=1.5)

    #         self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    #         env_origin_x = (i + 0.5) * self.env_length
    #         env_origin_y = (j + 0.5) * self.env_width
    #         x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
    #         x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
    #         y1 = int((self.env_width / 2.0 - 1) / self.horizontal_scale)
    #         y2 = int((self.env_width / 2.0 + 1) / self.horizontal_scale)
    #         env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
    #         self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

