import torch

import comfy.sd
import comfy.utils
import comfy.model_patcher
from comfy.supported_models import QwenImage
import comfy.model_management
import folder_paths

from .comfy_ops import block_scaled_fp8_ops


class MusubiUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": ([x for x in folder_paths.get_filename_list("diffusion_models") if "musubi" in x], ),
                "weight_dtype": (["default", "fp8_e5m2", "fp8_e4m3fn"], ),
            },
            "optional": {
                "block_size": ("INT", {"default": 64}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "musubi"

    def load_unet(self, unet_name, weight_dtype, block_size=16):
        model_options = {}
        dtype = None
        if weight_dtype == "fp8_e5m2":
            dtype = torch.float8_e5m2
        elif weight_dtype == "fp8_e4m3fn":
            dtype = torch.float8_e4m3fn
        model_options["dtype"] = dtype
        
        if block_size is not None and block_size > 1:
            model_options["custom_operations"] = block_scaled_fp8_ops(dtype, block_size)
        
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)