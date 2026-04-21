import torch


def get_free_device_indx(threshold_gb=4.0):
    if not torch.cuda.is_available():
        return -1, []

    free_mem = []
    for i in range(torch.cuda.device_count()):
        # try-except block to handle zero free memory to even initialize the device: would run into runtime error
        try:
            mem_free, total_mem = torch.cuda.mem_get_info(i)
            free_mem.append([mem_free / (2 ** 30), i])
        except:
            continue
    if not free_mem:
        return -1, []
    free_mem.sort(reverse=True)

    free_device_indx = -1 if free_mem[0][0] < threshold_gb else free_mem[0][1]
    device_indices = [i for mem, i in free_mem if mem >= threshold_gb]

    return free_device_indx, device_indices


def _validate_device_index(device_index):
    if device_index is None:
        return
    if not isinstance(device_index, int) or device_index < -1:
        raise ValueError("device_index must be None, -1, or a non-negative integer")


def get_inference_device_config(device_index=None, threshold_gb=4.0):
    """
    Resolve a requested device index into a canonical runtime configuration.

    `device_index == -1` always means CPU inference, even on hosts with CUDA.
    The returned config also normalizes library-specific device arguments so
    callers do not have to special-case CPU handling.
    """
    _validate_device_index(device_index)

    available_device_indices = []
    resolved_device_index = device_index
    if resolved_device_index is None:
        resolved_device_index, available_device_indices = get_free_device_indx(threshold_gb)

    use_cpu = resolved_device_index == -1 or not torch.cuda.is_available()
    if use_cpu:
        return {
            "requested_device_index": device_index,
            "resolved_device_index": resolved_device_index,
            "effective_device_index": -1,
            "available_device_indices": available_device_indices,
            "use_cpu": True,
            "device_str": "cpu",
            "torch_device": torch.device("cpu"),
            "whisperx_device": "cpu",
            "whisperx_device_index": 0,
            "sentence_transformer_device": "cpu",
            "transformers_pipeline_device": -1,
            "tf_device": "/CPU:0",
        }

    if resolved_device_index >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid device_index {resolved_device_index}: "
            f"only {torch.cuda.device_count()} GPU(s) detected"
        )

    device_str = f"cuda:{resolved_device_index}"
    return {
        "requested_device_index": device_index,
        "resolved_device_index": resolved_device_index,
        "effective_device_index": resolved_device_index,
        "available_device_indices": available_device_indices,
        "use_cpu": False,
        "device_str": device_str,
        "torch_device": torch.device(device_str),
        "whisperx_device": "cuda",
        "whisperx_device_index": resolved_device_index,
        "sentence_transformer_device": device_str,
        "transformers_pipeline_device": resolved_device_index,
        "tf_device": f"/GPU:{resolved_device_index}",
    }
