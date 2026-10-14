"""H3 pointwise reference rounding, expressed only in NineToothed."""
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

def _apply_for_arch(application, dtype, archs):
    import os
    if os.environ.get("H3_GFX906_PORTABLE_BF16") == "1" and dtype == ninetoothed.bfloat16:
        from ninetoothed.platform import current_target
        targets = archs or [current_target().arch]
        if any(target != "gfx906" for target in targets):
            return application
        from h3_gfx906_pointwise import APPLICATIONS
        return APPLICATIONS[application.__name__]
    return application



def swiglu_arrange(up, gate, out, rows=16, cols=256):
    arranged = []
    for tensor in (up, gate, out):
        tiled = tensor.tile((1, rows, cols))
        tiled.dtype = tiled.dtype.squeeze(0)
        arranged.append(tiled)
    return tuple(arranged)


def swiglu_apply(up, gate, out):
    value = gate
    value = value.to(ntl.float32)
    activated = (value / (1.0 + ntl.exp(-value))).to(gate.dtype)
    out = up * activated  # noqa: F841


def make_swiglu(caller="torch", output_dir=None, archs=None, rows=1, cols=1024, num_warps=4,
                dtype=ninetoothed.bfloat16):
    # Fixed row slices; tune their width and warp count on the target device.
    return ninetoothed.make(functools.partial(swiglu_arrange, rows=rows, cols=cols), _apply_for_arch(swiglu_apply, dtype, archs),
        tuple(Tensor(3, dtype=dtype) for _ in range(3)),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_swiglu_exact", num_warps=num_warps, num_stages=1)


def rms_arrange(x, weight, eps, out, block):
    a = x.tile((1, 1, block))
    a.dtype = a.dtype.squeeze((0, 1))
    w = weight.tile((block,)).unsqueeze(0).unsqueeze(0).expand(a.shape)
    o = out.tile((1, 1, block))
    o.dtype = o.dtype.squeeze((0, 1))
    return a, w, eps, o


def rms_apply(x, weight, eps, out):
    value = x
    value = value.to(ntl.float32)
    squared = (value * value).to(x.dtype).to(ntl.float32)
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.)
    mean = (ntl.sum(squared, 0) / x.source.shape[-1]).to(x.dtype)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32).to(x.dtype)
    variance = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(x.dtype)
    reciprocal = ntl.rsqrt(variance.to(ntl.float32)).to(x.dtype)
    normalized = (value * reciprocal.to(ntl.float32)).to(x.dtype)
    out = normalized * weight  # noqa: F841


def make_rms(caller="torch", output_dir=None, archs=None, width=5376,
             dtype=ninetoothed.bfloat16):
    shape = (None, None, width)
    return ninetoothed.make(functools.partial(rms_arrange, block=1 << (width - 1).bit_length()), _apply_for_arch(rms_apply, dtype, archs),
        (Tensor(shape=shape, dtype=dtype),
         Tensor(shape=(width,), dtype=dtype),
         Tensor(0, dtype=ninetoothed.float64),
         Tensor(shape=shape, dtype=dtype)),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_rms_exact" + (f"_{width}" if width else ""),
        num_warps=4, num_stages=1)


def rms_modulate_arrange(x, weight, eps, scale, shift, out):
    a, w, e, o = rms_arrange(x, weight, eps, out, block=8192)
    arranged = []
    for tensor in (scale, shift):
        t = tensor.tile((1, 1, 8192))
        t.dtype = t.dtype.squeeze((0, 1))
        arranged.append(t)
    return a, w, e, *arranged, o


def rms_modulate_apply(x, weight, eps, scale, shift, out):
    value = x
    value = value.to(ntl.float32)
    squared = (value * value).to(x.dtype).to(ntl.float32)
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.)
    mean = (ntl.sum(squared, 0) / x.source.shape[-1]).to(x.dtype)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32).to(x.dtype)
    variance = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(x.dtype)
    reciprocal = ntl.rsqrt(variance.to(ntl.float32)).to(x.dtype)
    normalized = (value * reciprocal.to(ntl.float32)).to(x.dtype)
    weighted = (normalized * weight).to(x.dtype)
    factor = (1. + scale).to(x.dtype)
    product = (weighted * factor).to(x.dtype)
    out = product + shift  # noqa: F841


def make_rms_modulate(caller="torch", output_dir=None, archs=None,
                      dtype=ninetoothed.bfloat16):
    def activation():
        return Tensor(shape=(None, None, 5376), dtype=dtype)
    return ninetoothed.make(rms_modulate_arrange, _apply_for_arch(rms_modulate_apply, dtype, archs),
        (activation(), Tensor(shape=(5376,), dtype=dtype),
         Tensor(0, dtype=ninetoothed.float64), activation(), activation(), activation()),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_rms_modulate_exact", num_warps=4, num_stages=1)



def gated_residual_arrange(x, gate, y, out, block=(16, 256)):
    arranged = []
    for tensor in (x, gate, y, out):
        tiled = tensor.tile((1, *block))
        tiled.dtype = tiled.dtype.squeeze(0)
        arranged.append(tiled)
    return tuple(arranged)


def gated_residual_apply(x, gate, y, out):
    # Reference order: BF16 product, then BF16 sum (two separate torch ops).
    product = (gate * y).to(x.dtype)
    out = x + product  # noqa: F841


def make_gated_residual(caller="torch", output_dir=None, archs=None, block=(4, 512), num_warps=8,
                        dtype=ninetoothed.bfloat16):
    # Pure elementwise: any tiling gives identical values; (4, 512)/8 warps is
    # retained as a baseline for target-specific measurement.
    def activation():
        return Tensor(shape=(None, None, 5376), dtype=dtype)
    return ninetoothed.make(functools.partial(gated_residual_arrange, block=block), _apply_for_arch(gated_residual_apply, dtype, archs),
        tuple(activation() for _ in range(4)),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_gated_residual_exact", num_warps=num_warps, num_stages=1)


def gated_residual_rms_modulate_arrange(x, gate, y, weight, eps, scale, shift, residual, out):
    a, w, e, o = rms_arrange(x, weight, eps, out, block=8192)
    arranged = []
    for tensor in (gate, y, scale, shift, residual):
        t = tensor.tile((1, 1, 8192))
        t.dtype = t.dtype.squeeze((0, 1))
        arranged.append(t)
    g, v, sc, sh, r = arranged
    return a, g, v, w, e, sc, sh, r, o


def gated_residual_rms_modulate_apply(x, gate, y, weight, eps, scale, shift, residual, out):
    # `residual = x + gate * y` exactly as `gated_residual_apply`, then the
    # same RMSNorm/AdaLN arithmetic as `rms_modulate_apply` on that residual.
    product = (gate * y).to(x.dtype)
    summed = (x + product).to(x.dtype)
    residual = summed  # noqa: F841
    value = summed.to(ntl.float32)
    squared = (value * value).to(x.dtype).to(ntl.float32)
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.)
    mean = (ntl.sum(squared, 0) / x.source.shape[-1]).to(x.dtype)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32).to(x.dtype)
    variance = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(x.dtype)
    reciprocal = ntl.rsqrt(variance.to(ntl.float32)).to(x.dtype)
    normalized = (value * reciprocal.to(ntl.float32)).to(x.dtype)
    weighted = (normalized * weight).to(x.dtype)
    factor = (1. + scale).to(x.dtype)
    modulated = (weighted * factor).to(x.dtype)
    out = modulated + shift  # noqa: F841


def make_gated_residual_rms_modulate(caller="torch", output_dir=None, archs=None, num_warps=4,
                                     dtype=ninetoothed.bfloat16):
    # Keep make_rms_modulate's 8192-wide rows and 4 warps: the FP32 sum of
    # squares then reduces in the same order, so the result is bitwise the
    # same as the unfused add + h3_rms_modulate.
    def activation():
        return Tensor(shape=(None, None, 5376), dtype=dtype)
    return ninetoothed.make(gated_residual_rms_modulate_arrange, _apply_for_arch(gated_residual_rms_modulate_apply, dtype, archs),
        (activation(), activation(), activation(), Tensor(shape=(5376,), dtype=dtype),
         Tensor(0, dtype=ninetoothed.float64), activation(), activation(), activation(), activation()),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_gated_residual_rms_modulate_exact", num_warps=num_warps, num_stages=1)

def rope_arrange(x0, x1, xp, cos, sin, y0, y1, yp):
    arranged = []
    for tensor in (x0[..., :48], x1[..., 48:96], xp[..., 96:],
                   y0[..., :48], y1[..., 48:96], yp[..., 96:]):
        t = tensor.tile((1, 1, 64))
        t.dtype = t.dtype.squeeze((0, 1))
        arranged.append(t)
    angles = []
    for tensor in (cos, sin):
        t = tensor.tile((1, 64))
        t.dtype = t.dtype.squeeze(0)
        angles.append(t.unsqueeze(1).expand((-1, arranged[0].shape[1], -1)))
    return *arranged[:3], *angles, *arranged[3:]


def rope_apply(x0, x1, passthrough, cos, sin, y0, y1, ypass):
    a = x0
    b = x1
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    ac = (a * c).to(x0.dtype).to(ntl.float32)
    bs = (-b * s).to(x0.dtype).to(ntl.float32)
    bc = (b * c).to(x0.dtype).to(ntl.float32)
    ass = (a * s).to(x0.dtype).to(ntl.float32)
    y0 = ac + bs  # noqa: F841
    y1 = bc + ass  # noqa: F841
    ypass = passthrough  # noqa: F841


def make_rope(caller="torch", output_dir=None, archs=None, dtype=ninetoothed.bfloat16):
    # This NineToothed version requires distinct arguments for multiple views
    # of one source. The caller aliases x0/x1/xp and y0/y1/yp respectively.
    def activation():
        return Tensor(shape=(None, None, 128), dtype=dtype)
    kernel = ninetoothed.make(rope_arrange, _apply_for_arch(rope_apply, dtype, archs),
        (activation(), activation(), activation(),
         Tensor(shape=(None, 48), dtype=dtype),
         Tensor(shape=(None, 48), dtype=dtype),
         activation(), activation(), activation()),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_rope_exact", num_warps=4, num_stages=1)
    def launch(x, cos, sin, out):
        return kernel(x, x, x, cos, sin, out, out, out)
    return launch


def qk_norm_rope_arrange(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp, rows=8,
                         block_tokens=1, ranks=1):
    # [T, P, Hs, 128] views; one program per (token block, rank block) handles
    # `block_tokens` tokens x `ranks` ranks x `rows` heads (block_tokens=1:
    # one token; ranks=1: one rank).
    if block_tokens == 1:
        squeezed = (0, 1) if ranks == 1 else (0,)
    else:
        squeezed = (1,)
    parts = []
    for tensor in (x0[..., :48], x1[..., 48:96], xp[..., 96:],
                   y0[..., :48], y1[..., 48:96], yp[..., 96:]):
        t = tensor.tile((block_tokens, ranks, rows, 64))
        t.dtype = t.dtype.squeeze(squeezed)
        parts.append(t)
    grid = parts[0].shape
    weights = []
    for tensor in (w0[:48], w1[48:96], wp[96:]):
        weights.append(tensor.tile((64,)).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(grid))
    angles = []
    for tensor in (cos, sin):
        t = tensor.tile((block_tokens, 64))
        if block_tokens == 1:
            t.dtype = t.dtype.squeeze(0)
        angles.append(t.unsqueeze(1).unsqueeze(1).expand((-1, grid[1], grid[2], -1)))
    return (*parts[:3], *weights, eps, *angles, *parts[3:])


def qk_norm_rope_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    # h3_rms_exact (width 128) followed by h3_rope_exact, per head row.
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    sa = (a * a).to(x0.dtype).to(ntl.float32)
    sb = (b * b).to(x0.dtype).to(ntl.float32)
    sp = (p * p).to(x0.dtype).to(ntl.float32)
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.)
    mean = ((ntl.sum(sa, 1) + ntl.sum(sb, 1) + ntl.sum(sp, 1)) / 128).to(x0.dtype)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32).to(x0.dtype)
    variance = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(x0.dtype)
    reciprocal = ntl.rsqrt(variance.to(ntl.float32)).to(x0.dtype).to(ntl.float32)
    na = ((a * reciprocal[:, None]).to(x0.dtype) * w0).to(x0.dtype).to(ntl.float32)
    nb = ((b * reciprocal[:, None]).to(x0.dtype) * w1).to(x0.dtype).to(ntl.float32)
    npass = ((p * reciprocal[:, None]).to(x0.dtype) * wp).to(x0.dtype)
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    ac = (na * c).to(x0.dtype).to(ntl.float32)
    bs = (-nb * s).to(x0.dtype).to(ntl.float32)
    bc = (nb * c).to(x0.dtype).to(ntl.float32)
    ass = (na * s).to(x0.dtype).to(ntl.float32)
    y0 = ac + bs  # noqa: F841
    y1 = bc + ass  # noqa: F841
    yp = npass  # noqa: F841


def qk_norm_rope_ranks_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    # Same arithmetic as qk_norm_rope_apply on a [ranks, heads, 64] block of
    # one token: per-head reductions over the last axis, angles broadcast.
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    sa = (a * a).to(x0.dtype).to(ntl.float32)
    sb = (b * b).to(x0.dtype).to(ntl.float32)
    sp = (p * p).to(x0.dtype).to(ntl.float32)
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.)
    mean = ((ntl.sum(sa, 2) + ntl.sum(sb, 2) + ntl.sum(sp, 2)) / 128).to(x0.dtype)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32).to(x0.dtype)
    variance = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(x0.dtype)
    reciprocal = ntl.rsqrt(variance.to(ntl.float32)).to(x0.dtype).to(ntl.float32)
    na = ((a * reciprocal[:, :, None]).to(x0.dtype) * w0).to(x0.dtype).to(ntl.float32)
    nb = ((b * reciprocal[:, :, None]).to(x0.dtype) * w1).to(x0.dtype).to(ntl.float32)
    npass = ((p * reciprocal[:, :, None]).to(x0.dtype) * wp).to(x0.dtype)
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    ac = (na * c).to(x0.dtype).to(ntl.float32)
    bs = (-nb * s).to(x0.dtype).to(ntl.float32)
    bc = (nb * c).to(x0.dtype).to(ntl.float32)
    ass = (na * s).to(x0.dtype).to(ntl.float32)
    y0 = ac + bs  # noqa: F841
    y1 = bc + ass  # noqa: F841
    yp = npass  # noqa: F841


def qk_norm_rope_block_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    # Same arithmetic as qk_norm_rope_apply on a [tokens, heads, 64] block:
    # the per-head reductions run over the last axis and the per-token angles
    # broadcast over the heads.
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    sa = (a * a).to(x0.dtype).to(ntl.float32)
    sb = (b * b).to(x0.dtype).to(ntl.float32)
    sp = (p * p).to(x0.dtype).to(ntl.float32)
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.)
    mean = ((ntl.sum(sa, 2) + ntl.sum(sb, 2) + ntl.sum(sp, 2)) / 128).to(x0.dtype)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32).to(x0.dtype)
    variance = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(x0.dtype)
    reciprocal = ntl.rsqrt(variance.to(ntl.float32)).to(x0.dtype).to(ntl.float32)
    na = ((a * reciprocal[:, :, None]).to(x0.dtype) * w0).to(x0.dtype).to(ntl.float32)
    nb = ((b * reciprocal[:, :, None]).to(x0.dtype) * w1).to(x0.dtype).to(ntl.float32)
    npass = ((p * reciprocal[:, :, None]).to(x0.dtype) * wp).to(x0.dtype)
    c = cos
    s = sin
    c = c.to(ntl.float32)[:, None, :]
    s = s.to(ntl.float32)[:, None, :]
    ac = (na * c).to(x0.dtype).to(ntl.float32)
    bs = (-nb * s).to(x0.dtype).to(ntl.float32)
    bc = (nb * c).to(x0.dtype).to(ntl.float32)
    ass = (na * s).to(x0.dtype).to(ntl.float32)
    y0 = ac + bs  # noqa: F841
    y1 = bc + ass  # noqa: F841
    yp = npass  # noqa: F841


def make_qk_norm_rope(caller="torch", output_dir=None, archs=None, rows=8, num_warps=1,
                      block_tokens=1, ranks=1, dtype=ninetoothed.bfloat16):
    """Per-head RMSNorm (weight [128]) + 3-axis RoPE on [T, P, Hs, 128] (any
    strides), written to another [T, P, Hs, 128] view (e.g. an all-to-all
    send buffer). Arguments: x, weight, eps, cos, sin ([T, 48]), out.
    ``block_tokens`` tokens x ``rows`` heads per program. Block shape and
    warp count may be measured with the operator benchmark for each target.
    """
    def view4():
        return Tensor(shape=(None, None, None, 128), dtype=dtype)
    def weight():
        return Tensor(shape=(128,), dtype=dtype)
    if block_tokens == 1:
        apply = qk_norm_rope_apply if ranks == 1 else qk_norm_rope_ranks_apply
    elif ranks == 1:
        apply = qk_norm_rope_block_apply
    else:
        raise ValueError("qk_norm_rope: block_tokens > 1 needs ranks == 1")
    kernel = ninetoothed.make(functools.partial(qk_norm_rope_arrange, rows=rows,
                                                block_tokens=block_tokens, ranks=ranks), _apply_for_arch(apply, dtype, archs),
        (view4(), view4(), view4(), weight(), weight(), weight(),
         Tensor(0, dtype=ninetoothed.float64),
         Tensor(shape=(None, 48), dtype=dtype),
         Tensor(shape=(None, 48), dtype=dtype),
         view4(), view4(), view4()),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_qk_norm_rope", num_warps=num_warps, num_stages=1)
    if caller != "torch":
        return kernel
    def launch(x, weight, eps, cos, sin, out):
        return kernel(x, x, x, weight, weight, weight, eps, cos, sin, out, out, out)
    return launch


def rope_cache_arrange(position, power, cos, sin, block_tokens=64):
    # position [T, 3] (FP64 as the pipeline passes it); power [16] FP32;
    # cos/sin [T, 48] (the two halves of the [T, 96] cache). One program per
    # (token block, axis): `block_tokens` positions of one axis and the 16
    # frequencies of that axis.
    cos = cos.tile((block_tokens, 16))
    sin = sin.tile((block_tokens, 16))
    power = power.tile((16,)).unsqueeze(0).expand(cos.shape)
    return position.tile((block_tokens, 1)), power, cos, sin


def rope_cache_apply(position, power, cos, sin):
    # minimax_h3_rotary_cache: inv_freq = 1 / power with power = theta **
    # (arange(0, 32, 2) / 32) (InfiniCore float_power, the same device powf as
    # PyTorch), angles = position.float() * inv_freq, cat(cos, sin) rounded to
    # BF16; cos/sin are the same ocml functions as the PyTorch ops.
    p = power
    inverse = 1. / p
    x = position
    x = x.to(ntl.float32)
    angle = x * inverse[None, :]
    cos = libdevice.cos(angle).to(ntl.bfloat16)  # noqa: F841
    sin = libdevice.sin(angle).to(ntl.bfloat16)  # noqa: F841


def make_rope_cache(caller="torch", output_dir=None, archs=None, block_tokens=64, num_warps=4,
                    dtype=ninetoothed.bfloat16):
    """H3's 3-axis RoPE table (rope_freq_dim 16): position [T, 3] FP64 and the
    16 powers theta ** (2k / 32) (FP32) to cos/sin [T, 48] BF16 views (the
    halves of a [T, 96] cache), bitwise equal to minimax_h3_rotary_cache."""
    def table():
        return Tensor(shape=(None, 48), dtype=dtype)
    return ninetoothed.make(functools.partial(rope_cache_arrange, block_tokens=block_tokens),
        rope_cache_apply,
        (Tensor(shape=(None, 3), dtype=ninetoothed.float64), Tensor(shape=(16,), dtype=ninetoothed.float32),
         table(), table()),
        caller=caller, output_dir=output_dir, archs=archs,
        kernel_name="h3_rope_cache", num_warps=num_warps, num_stages=1)
