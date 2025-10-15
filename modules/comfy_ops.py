import logging

import torch

import comfy.float
from comfy.ops import cast_bias_weight, manual_cast


def quantize_block_weight(weight, scale_weight):
    if scale_weight.ndim < 3:
        quantized_weight = weight / scale_weight
    else:
        out_features, num_blocks, _ = scale_weight.shape
        quantized_weight = weight.view(out_features, num_blocks, -1) / scale_weight
        quantized_weight = quantized_weight.view_as(weight)
    return quantized_weight

def dequantize_block_weight(weight, scale_weight):
    if scale_weight.ndim < 3:
        dequantized_weight = weight * scale_weight
    else:
        out_features, num_blocks, _ = scale_weight.shape
        dequantized_weight = weight.view(out_features, num_blocks, -1) * scale_weight
        dequantized_weight = dequantized_weight.view_as(weight)
    return dequantized_weight

def block_scaled_fp8_ops(override_dtype=None, block_size=64):
    logging.info(f"Using block scaled fp8: fp8 dtype: {override_dtype}, block size: {block_size}")

    class block_scaled_fp8_op(manual_cast):
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                # override Linear dtype if specified
                if override_dtype is not None:
                    kwargs['dtype'] = override_dtype
                super().__init__(*args, **kwargs)

            # override Linear's reset_parameters
            def reset_parameters(self):
                # add scale_weight parameter to load scale factor for each block
                if not hasattr(self, 'scale_weight'):
                    out_features, in_features = self.weight.shape
                    if in_features % block_size != 0:
                        scale_shape = (out_features, in_features)
                    else:
                        scale_shape = (out_features, in_features // block_size, 1)
                    self.scale_weight = torch.nn.parameter.Parameter(
                        data=torch.ones(scale_shape, device=self.weight.device, dtype=torch.float32),
                        requires_grad=False,
                    )

                return None

            # override manual_cast.Linear's forward_comfy_cast_weights
            def forward_comfy_cast_weights(self, input):
                weight, bias = cast_bias_weight(self, input)
                device, dtype = weight.device, weight.dtype
                self.scale_weight.data = self.scale_weight.data.to(device=device, dtype=dtype)

                return torch.nn.functional.linear(input, dequantize_block_weight(weight, self.scale_weight), bias)

            # comfyui load lora will call convert_weight
            def convert_weight(self, weight, inplace=False, **kwargs):
                scale_weight = self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                if inplace:
                    weight = dequantize_block_weight(weight, scale_weight)
                    return weight
                else:
                    return dequantize_block_weight(weight, scale_weight)

            # comfyui load lora will call set_weight
            def set_weight(self, weight, inplace_update=False, seed=None, **kwargs):
                scale_weight = self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                weight = comfy.float.stochastic_rounding(quantize_block_weight(weight, scale_weight), self.weight.dtype, seed=seed)
                if inplace_update:
                    self.weight.data.copy_(weight)
                else:
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)

    return block_scaled_fp8_op
