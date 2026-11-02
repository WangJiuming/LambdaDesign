"""Strict checkpoint loading; weights are supplied by the caller."""

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
import hashlib
import math
import re

import torch

from .model import LatentDesignModel
from .molclr import MolCLREncoder
from .vocabulary import SelfiesVocabulary

MODEL_KEYS = ("node_dim", "d_model", "nhead", "enc_layers", "dec_layers",
              "dim_feedforward", "latent_dim", "hidden_dim", "encoder_pool")

# Taken from the original training labels, not estimated from inference
# inputs. Bind the transformation to the verified weight file so another model
# cannot silently inherit the HeLa/mRNA label scale.
VERIFIED_ACTIVITY_TRANSFORMS = {
    "51cc54bf08f867506013fd28c9394a1bf6c1e4630eae86559e856ffcbeb4196c": {
        "scale": 7.826270046,
        "offset": -2.621690952,
        "output_scale": "original HeLa/mRNA Experiment_value",
        "source": "verified original training-label min-max normalization",
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weights(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Weights are supplied separately; see README.md.")
    # Do not silently fall back to unrestricted pickle for a legacy checkpoint.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Cannot safely load {path.name}: {error}. Use the safe-loading checkpoint described in README.md.") from error


def resolve_device(request):
    if request == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu, cuda:N and auto")
    if device.type == "cuda" and (not torch.cuda.is_available() or
            (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError(f"Requested device {request} is unavailable; use --device cpu")
    return device


@dataclass
class DesignCheckpoint:
    model: LatentDesignModel
    vocabulary: SelfiesVocabulary
    activity_statistics: dict
    architecture: dict
    sha256: str
    activity_transform: dict | None = None


def load_design_checkpoint(path, device):
    saved = load_weights(path)
    if not isinstance(saved, Mapping):
        raise ValueError("Expected a LambdaDesign checkpoint dictionary")
    required = {"model_state", "vocab_stoi", "cvae_cfg", "data_cfg"}
    if not required.issubset(saved):
        raise ValueError("Expected the HeLa/mRNA shared-latent checkpoint with model_state, vocab_stoi, cvae_cfg and data_cfg")
    vocabulary = SelfiesVocabulary(saved["vocab_stoi"])
    try:
        config = {key:saved["cvae_cfg"][key] for key in MODEL_KEYS}
        config.update({key:saved["data_cfg"][key] for key in ("max_atoms", "max_seq_len")})
    except (KeyError, TypeError) as error:
        raise ValueError(f"Incomplete architecture metadata: {error}") from error
    if any(type(v) is not int or v <= 0 for k,v in config.items() if k != "encoder_pool"):
        raise ValueError("Checkpoint dimensions must be positive integers")
    if config["encoder_pool"] not in ("mean", "cls") or config["d_model"] % config["nhead"] or config["d_model"] % 2:
        raise ValueError("Unsupported encoder pooling or attention dimensions")
    model = LatentDesignModel(vocab_size=len(vocabulary), **config)
    model.load_state_dict(saved["model_state"], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
        raise ValueError("Checkpoint contains non-finite model weights")
    model.requires_grad_(False).eval().to(device)
    statistics = {}
    for key in ("mean", "std", "p75", "p90", "p95"):
        if "act_" + key in saved:
            value = float(saved["act_" + key])
            if not math.isfinite(value):
                raise ValueError(f"Checkpoint activity statistic {key} is not finite")
            statistics[key] = value
    digest = file_sha256(path)
    transform = VERIFIED_ACTIVITY_TRANSFORMS.get(digest)
    return DesignCheckpoint(model, vocabulary, statistics, config, digest,
                            dict(transform) if transform is not None else None)


def load_molclr_checkpoint(path, device, expected_width):
    saved = load_weights(path)
    if not isinstance(saved, Mapping):
        raise ValueError("Expected a MolCLR state dictionary")
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in saved:
            saved = saved[key]
            break
    if not isinstance(saved, Mapping) or not saved:
        raise ValueError("MolCLR checkpoint has no state dictionary")
    state = dict(saved)
    for prefix in ("module.", "model.", "encoder."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]:v for k,v in state.items()}
    if "x_embedding1.weight" not in state:
        raise ValueError("MolCLR atom embeddings are missing")
    width = state["x_embedding1.weight"].shape[1]
    layers = sorted({int(m.group(1)) for k in state if (m := re.match(r"gnns\.(\d+)\.", k))})
    if not layers or layers != list(range(len(layers))) or width != expected_width:
        raise ValueError("MolCLR layer layout or feature width does not match the design model")
    model = MolCLREncoder(width, len(layers))
    # Only the documented, unused graph-level projection is excluded.
    core = {k:v for k,v in state.items() if not k.startswith(("feat_lin.", "out_lin."))}
    model.load_state_dict(core, strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
        raise ValueError("MolCLR checkpoint contains non-finite weights")
    model.requires_grad_(False).eval().to(device)
    return model
