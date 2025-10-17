import torch

import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_detection
import folder_paths

from .comfy_ops import block_scaled_fp8_ops
from .lora_utils import load_safetensors_with_lora_and_fp8
from .fp8_optimization_utils import optimize_state_dict_with_fp8

dtypes = {
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e4m3fnuz": torch.float8_e4m3fnuz,
}

class MusubiUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": ([x for x in folder_paths.get_filename_list("diffusion_models") if "musubi" in x], ),
                "weight_dtype": (list(dtypes.keys()), ),
            },
            "optional": {
                "block_size": ("INT", {"default": 64}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "musubi"

    def load_unet(self, unet_name, weight_dtype, block_size=64):
        # load weights
        unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)
        target_dtype = dtypes[weight_dtype]

        model_options = {
            "custom_operations": block_scaled_fp8_ops(target_dtype, block_size),
        }

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        return (model,)


class MusubiUNETJitLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "weight_dtype": (list(dtypes.keys()), ),
            },
            "optional": {
                "block_size": ("INT", {"default": 64}),
                "lora_name": (["none"] + folder_paths.get_filename_list("loras"), ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "musubi"

    def load_unet(self, unet_name, lora_name, weight_dtype, block_size=64):
        lora_name = None if lora_name == "none" else lora_name
        target_dtype = dtypes[weight_dtype]

        # load weights
        unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)

        model_options = {
            "custom_operations": block_scaled_fp8_ops(target_dtype, block_size),
        }

        lora_weights = []
        if lora_name:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora_weights = [comfy.utils.load_torch_file(lora_path)]

        sd = load_safetensors_with_lora_and_fp8(unet_path, lora_weights, lora_multipliers=None, fp8_optimization=False, calc_device=None)

        # fp8 optimization
        if target_dtype == torch.float8_e5m2:
            sd = optimize_state_dict_with_fp8(sd, exp_bits=5, mantissa_bits=2, block_size=block_size, calc_device=None)
        elif target_dtype == torch.float8_e4m3fn:
            sd = optimize_state_dict_with_fp8(sd, exp_bits=4, mantissa_bits=3, block_size=block_size, calc_device=None)
        else:
            raise Exception(f"Unsupported weight_dtype: {weight_dtype}")

        # fit model to comfy.ModelPatcher
        model_config = comfy.model_detection.model_config_from_unet(sd, "")
        # FIXME: manual_cast_type should be set according to origin model dtype
        model_config.set_inference_dtype(target_dtype, torch.bfloat16)
        model_config.custom_operations = model_options.get("custom_operations", None)

        model = model_config.get_model(sd, "")
        model.load_model_weights(sd, "")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        model = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

        return (model,)
