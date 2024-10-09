# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.                                                    #                                                                                                                       
# NVIDIA CORPORATION and its licensors retain all intellectual property                                                 
# and proprietary rights in and to this software, related documentation                                                 
# and any modifications thereto.  Any use, reproduction, disclosure or                                                  
# distribution of this software and related documentation without an express                                            
# license agreement from NVIDIA CORPORATION is strictly prohibited.                                                                                                                                                                             

from omni.isaac.kit import SimulationApp 
import omni                                                                                                                                                                                                       
# This sample enables a livestream server to connect to when running headless                                           
CONFIG = {                                                                                                                  
    "width": 1280,                                                                                                          
    "height": 720,                                                                                                          
    "window_width": 1920,                                                                                                   
    "window_height": 1080,                                                                                                  
    "headless": True, 
    "enable_livestream": True,                                                                                                      
    "renderer": "RayTracedLighting",                                                                                        
    "display_options": 3286, 
     
}                                                                                                                                                                                                                                                                                                                                                                      
# Start the omniverse application                                                                                       
kit = SimulationApp(launch_config=CONFIG)                                                                                                                                                                                                       
from omni.isaac.core.utils.extensions import enable_extension   
from omni.isaac.core import World                                                                                                                                                                                
# Enable Livestream extension                                                                                           
kit.set_setting("/app/window/drawMouse", True)                                                                        
kit.set_setting("/app/livestream/proto", "ws")                                                                        
kit.set_setting("/app/livestream/websocket/framerate_limit", 120) 
kit.set_setting("/ngx/enabled", False)                                                  
                                                                      
enable_extension("omni.kit.livestream.native")                                                                                                                                                                                                
                                                                                    
omni.usd.get_context().open_stage("omniverse://10.10.51.5/NVIDIA/Assets/Isaac/2022.2.1/Isaac/Environments/Hospital/hospital.usd")

while kit.is_running():
    kit.update()

kit.close()
