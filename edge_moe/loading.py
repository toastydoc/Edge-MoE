"""Load Gemma 4 MoE with the routed experts streamed from disk.

The 26B checkpoint is 51.6GB in bf16, and the routed experts (45.7GB) are 3D
nn.Parameters that bitsandbytes will not quantise, so the model cannot sit in
32GB RAM + 16GB VRAM. Instead:

  * everything except the routed experts (~6GB bf16) lives on the GPU;
  * each layer's `experts` weights (1.5GB) are read from the safetensors file
    onto the GPU just before that layer runs and freed straight after.

Every forward pass therefore re-reads the expert weights, so the cost is per
batch, not per token: use large batches and single-pass scoring, not generate().

Tensors are read with plain file I/O rather than safetensors.safe_open /
from_pretrained: on Windows opening the 50GB shard reserves commit charge for
the whole file and fails with "paging file is too small" (os error 1455).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "google/gemma-4-26B-A4B-it"
EXPERTS_CLASS_NAME = "Gemma4TextExperts"

# safetensors dtype -> (numpy dtype to read raw bytes as, torch dtype to view them as)
_DTYPES = {
    "BF16": (np.uint16, torch.bfloat16),
    "F16": (np.float16, torch.float16),
    "F32": (np.float32, torch.float32),
    "I64": (np.int64, torch.int64),
}


class SafetensorsIndex:
    """Name -> (file, offset, shape, dtype) over every shard in a directory."""

    def __init__(self, model_dir: str | Path):
        self.entries: dict[str, tuple[Path, int, tuple[int, ...], str]] = {}
        files = sorted(Path(model_dir).glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {model_dir}")
        for file in files:
            with open(file, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len))
            header.pop("__metadata__", None)
            for name, info in header.items():
                self.entries[name] = (file, 8 + header_len + info["data_offsets"][0], tuple(info["shape"]), info["dtype"])

    def keys(self):
        return self.entries.keys()

    def read(self, name: str, expert_ids: np.ndarray | None = None) -> torch.Tensor:
        """Read one tensor to CPU. `expert_ids` reads only those rows of dim 0."""
        file, offset, shape, dtype = self.entries[name]
        np_dtype, torch_dtype = _DTYPES[dtype]
        row = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        if expert_ids is None:
            arr = np.fromfile(file, dtype=np_dtype, count=int(np.prod(shape, dtype=np.int64)), offset=offset)
            out_shape = shape
        else:
            arr = np.empty((len(expert_ids), row), dtype=np_dtype)
            with open(file, "rb") as f:
                for i, e in enumerate(expert_ids):
                    f.seek(offset + int(e) * row * arr.itemsize)
                    f.readinto(arr[i])
            out_shape = (len(expert_ids), *shape[1:])
        return torch.from_numpy(arr).view(torch_dtype).reshape(out_shape)


def _model_class(config):
    return AutoModelForCausalLM if config.model_type == "gemma4_text" else AutoModelForImageTextToText


def resolve_model_dir(model_id: str) -> Path:
    if Path(model_id).is_dir():
        return Path(model_id)
    return Path(snapshot_download(model_id))  # no-op if already cached


def load_streaming(model_id: str = MODEL_ID, device: str | int = 0, dtype=torch.bfloat16):
    """Returns (model, tokenizer). Model is in eval mode with experts read per forward pass."""
    device = torch.device(device if isinstance(device, str) else f"cuda:{device}")
    model_dir = resolve_model_dir(model_id)
    config = AutoConfig.from_pretrained(model_dir)
    index = SafetensorsIndex(model_dir)

    with init_empty_weights():
        model = _model_class(config).from_config(config)
    model_keys = set(model.state_dict().keys())
    unexpected = sorted(set(index.keys()) - model_keys)
    if unexpected:
        raise RuntimeError(f"checkpoint tensors with no home in the model: {unexpected[:5]} (+{len(unexpected) - 5})")

    for name in index.keys():
        if ".experts." not in name:
            set_module_tensor_to_device(model, name, device, value=index.read(name), dtype=dtype)
    model.tie_weights()

    # Anything still on meta apart from the experts was missing from the checkpoint.
    missing = [n for n, t in model.state_dict().items() if t.is_meta and ".experts." not in n]
    if missing:
        raise RuntimeError(f"model tensors not found in checkpoint: {missing[:5]}")
    for name, buf in list(model.named_buffers()):  # non-persistent buffers built on CPU
        if buf.device != device:
            set_module_tensor_to_device(model, name, device, value=buf)

    for name, module in model.named_modules():
        if type(module).__name__ == EXPERTS_CLASS_NAME:
            _attach_expert_streaming(module, name, index, device, dtype)

    # The default "grouped_mm" experts path used ~10GB of extra VRAM on a 5k-token
    # batch; the eager per-expert loop is far lighter and speed is disk-bound anyway.
    model.set_experts_implementation("eager")
    model.eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.padding_side = "right"
    except (OSError, ValueError):
        tokenizer = None  # weight-only checkpoints (tests)
    return model, tokenizer


def _attach_expert_streaming(module: nn.Module, name: str, index: SafetensorsIndex, device, dtype) -> None:
    param_names = [n for n, _ in module.named_parameters(recurse=False)]
    placeholders = {n: module._parameters[n] for n in param_names}

    def load(mod, args):
        for n in param_names:
            t = index.read(f"{name}.{n}").to(device=device, dtype=dtype)
            mod._parameters[n] = nn.Parameter(t, requires_grad=False)

    def unload(mod, args, output):
        for n in param_names:
            mod._parameters[n] = placeholders[n]

    module.register_forward_pre_hook(load)
    module.register_forward_hook(unload, always_call=True)


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device
