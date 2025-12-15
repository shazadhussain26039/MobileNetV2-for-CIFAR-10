import torch
import torch.nn as nn
import math
from copy import deepcopy

# =====================
# Per-tensor quantization (used for conv weights + activations)
# =====================

def calc_scale_symmetric(tensor: torch.Tensor, num_bits: int):
    """
    Per-tensor symmetric quantization.
    scale = max(|x|) / qmax, where qmax = 2^{b-1}-1.
    """
    qmax = 2 ** (num_bits - 1) - 1
    max_val = tensor.abs().max()
    if max_val == 0:
        scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
    else:
        scale = max_val / qmax
    return scale, qmax


def quantize_tensor(tensor: torch.Tensor, num_bits: int):
    """
    Per-tensor quantization.
    Returns:
        q_tensor: int16 tensor
        scale: scalar tensor
    """
    scale, qmax = calc_scale_symmetric(tensor, num_bits)
    q = tensor / scale
    q = torch.clamp(q, -qmax - 1, qmax)
    q = q.round()
    return q.to(torch.int16), scale


def dequantize_tensor(q_tensor: torch.Tensor, scale: torch.Tensor):
    """
    Dequantize per-tensor quantized values.
    """
    return q_tensor.float() * scale


# =====================
# Per-channel quantization for Linear weights only
# =====================

def calc_scale_symmetric_per_channel_linear(tensor: torch.Tensor, num_bits: int):
    """
    Per-channel symmetric quantization for Linear weights.
    tensor: [out_features, in_features]
    We quantize per output feature (dim=0).
    """
    qmax = 2 ** (num_bits - 1) - 1
    max_vals = tensor.abs().amax(dim=1, keepdim=True)  # per-row max
    max_vals[max_vals == 0] = 1e-8
    scale = max_vals / qmax  # shape [out_features, 1]
    return scale, qmax


def quantize_linear_per_channel(tensor: torch.Tensor, num_bits: int):
    """
    Per-channel quantization for Linear weights.
    Returns:
        q_tensor: int16 tensor, same shape
        scale: float tensor of shape [out_features]
    """
    scale, qmax = calc_scale_symmetric_per_channel_linear(tensor, num_bits)
    q = tensor / scale
    q = torch.clamp(q, -qmax - 1, qmax)
    q = q.round()
    return q.to(torch.int16), scale.squeeze()  # scale: [out_features]


# =====================
# Quantized wrapper: Conv2d / Linear
# =====================

class QuantizedModuleWrapper(nn.Module):
    """
    Wraps Conv2d or Linear with quantized weights:
      - Conv2d: per-tensor weights
      - Linear: per-channel weights (per output feature)
    Activations: per-tensor quantization (simple and robust).
    """

    def __init__(self, module: nn.Module, weight_bits: int, activation_bits: int = 0):
        super().__init__()
        assert isinstance(module, (nn.Conv2d, nn.Linear))

        self.module_type = type(module)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

        # ----- Weight quantization -----
        if isinstance(module, nn.Conv2d):
            # Per-tensor quantization for conv weights
            w = module.weight.data
            q_w, w_scale = quantize_tensor(w, weight_bits)  # scalar scale
            self.register_buffer("q_weight", q_w)
            self.register_buffer("w_scale", w_scale)       # scalar

        else:  # Linear
            w = module.weight.data
            q_w, w_scale = quantize_linear_per_channel(w, weight_bits)
            self.register_buffer("q_weight", q_w)
            self.register_buffer("w_scale", w_scale)       # [out_features]

        # Bias in fp32
        if module.bias is not None:
            self.bias = nn.Parameter(module.bias.data.clone())
        else:
            self.bias = None

        # Conv2d hyperparams
        if isinstance(module, nn.Conv2d):
            self.stride = module.stride
            self.padding = module.padding
            self.dilation = module.dilation
            self.groups = module.groups

        # Metadata (for overhead accounting)
        self.register_buffer("meta_weight_bits", torch.tensor(weight_bits, dtype=torch.int32))
        self.register_buffer("meta_activation_bits", torch.tensor(activation_bits, dtype=torch.int32))

    def forward(self, x):
        # ----- Activation quantization: per-tensor only -----
        if self.activation_bits > 0:
            q_x, a_scale = quantize_tensor(x, self.activation_bits)
            x = dequantize_tensor(q_x, a_scale)

        # ----- Dequantize weights -----
        if self.module_type is nn.Conv2d:
            # per-tensor scale (scalar)
            w = dequantize_tensor(self.q_weight, self.w_scale)
            return nn.functional.conv2d(
                x,
                w,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            # Linear: per-output-feature scale [out_features]
            scale = self.w_scale.view(-1, 1)          # [out_features, 1]
            w = self.q_weight.float() * scale         # broadcast over in_features
            return nn.functional.linear(x, w, self.bias)


# =====================
# Model-wide quantization
# =====================

def quantize_model(model: nn.Module, weight_bits: int, activation_bits: int = 0, modules_to_ignore=None):
    """
    Deepcopy the model and replace Conv2d / Linear with QuantizedModuleWrapper.
    modules_to_ignore: list of names or types to skip (e.g., final classifier).
    """
    if modules_to_ignore is None:
        modules_to_ignore = []

    q_model = deepcopy(model)

    for name, module in q_model.named_children():
        # Recurse into children
        if len(list(module.children())) > 0:
            setattr(
                q_model,
                name,
                quantize_model(module, weight_bits, activation_bits, modules_to_ignore),
            )

        # Replace leaf modules
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            ignore = False
            for ig in modules_to_ignore:
                if isinstance(ig, str) and ig in name:
                    ignore = True
                if isinstance(ig, type) and isinstance(module, ig):
                    ignore = True
            if not ignore:
                setattr(
                    q_model,
                    name,
                    QuantizedModuleWrapper(
                        module,
                        weight_bits=weight_bits,
                        activation_bits=activation_bits,
                    ),
                )

    return q_model


# =====================
# Size / compression metrics
# =====================

def model_size_bytes_fp32(model: nn.Module):
    """
    Approximate fp32 model size: all parameters as 4-byte floats.
    """
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    return total_params * 4


def model_size_bytes_quantized(model: nn.Module):
    """
    Approximate quantized model size in bytes:
    - quantized weights with given bit-width
    - fp32 bias and BN params
    - weight scales and simple metadata
    """
    total_bits = 0

    for module in model.modules():
        if isinstance(module, QuantizedModuleWrapper):
            # weights
            numel_w = module.q_weight.numel()
            total_bits += numel_w * module.weight_bits

            # bias in fp32
            if module.bias is not None:
                total_bits += module.bias.numel() * 32

            # scales:
            #   - Conv2d: scalar scale
            #   - Linear: [out_features]
            total_bits += module.w_scale.numel() * 32

            # metadata (simple)
            total_bits += 32  # weight_bits
            total_bits += 32  # activation_bits

        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            # any Conv/Linear not quantized: count fp32
            for p in module.parameters(recurse=False):
                total_bits += p.numel() * 32

    # BatchNorm etc.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for p in module.parameters(recurse=False):
                total_bits += p.numel() * 32
            if hasattr(module, "running_mean") and module.running_mean is not None:
                total_bits += module.running_mean.numel() * 32
            if hasattr(module, "running_var") and module.running_var is not None:
                total_bits += module.running_var.numel() * 32

    total_bytes = math.ceil(total_bits / 8.0)
    return total_bytes


def calc_compression_ratio(original_bytes: int, compressed_bytes: int):
    """
    Simple compression ratio: original_size / compressed_size.
    """
    return original_bytes / compressed_bytes
