from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List


def create_profiling_state() -> Dict[str, Any]:
    """Create a nested dictionary used to accumulate profiling samples."""
    return {
        "denoising_step_ms": defaultdict(list),      # step_index -> [durations]
        "add_noise_step_ms": defaultdict(list),      # step_index -> [durations]
        "attention_kernel_ms": {"self": [], "cross": []},
        "attention_kernel_counts": {"self": 0, "cross": 0},
        "kv_prepare_ms": [],
        "kv_apply_ms": [],
        "kv_sink_concat_ms": [],
        "kv_cache_op_counts": {"prepare": 0, "apply": 0, "sink_concat": 0},
        "kv_recache_ms": {"gpu": [], "cpu": [], "host_transfer": []},
        "vae_decode_ms": [],
        "vae_encode_ms": [],
        "quantize_ms": [],
        "dequantize_ms": [],
        "host_sync_ms": defaultdict(list),            # label -> [durations]
        "device_transfer_ms": defaultdict(list),      # label -> [durations]
    }


def append_timing(bucket: List[float], value_ms: float) -> None:
    bucket.append(float(value_ms))


def append_labeled_timing(bucket: DefaultDict[str, List[float]], label: str, value_ms: float) -> None:
    bucket[label].append(float(value_ms))
