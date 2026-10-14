"""Runtime platform discovery and validated backend compiler options."""
import dataclasses


def current_target():
    import triton
    if __import__("os").environ.get("H3_DTK_TRITON_COMPAT") == "1":
        from dtk_triton_compat import install
        install()
    return triton.runtime.driver.active.get_current_target()


def backend_options_for_target(options, target=None):
    if not options:
        return {}
    from triton.compiler.compiler import make_backend
    target = target or current_target()
    # Hygon tuning knobs have no meaning on the NVIDIA compiler.
    if target.backend != "hip":
        return {}
    parsed = make_backend(target).parse_options({})
    supported = {f.name for f in dataclasses.fields(parsed)}
    unknown = set(options) - supported
    if unknown:
        raise ValueError(f"Unsupported {target.backend}/{target.arch} options: {sorted(unknown)}")
    return dict(options)


def profile():
    target = current_target()
    return {"backend": target.backend, "arch": target.arch,
            "warp_size": target.warp_size, "num_warps": 4, "num_stages": 1,
            "execution_modes": ["jit", "aot"]}
