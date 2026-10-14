import triton


def calculate_default_configs():
    from ninetoothed.platform import current_target
    current_target()  # Apply any explicitly enabled DTK metadata adapter first.
    device = triton.runtime.driver.active.get_current_device()
    properties = triton.runtime.driver.active.utils.get_device_properties(device)
    max_shared_mem = properties["max_shared_mem"]

    num_warps = 8
    num_stages = max_shared_mem // 2**15

    return num_warps, num_stages
