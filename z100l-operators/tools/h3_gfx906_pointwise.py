"""NineToothed BF16 arithmetic for gfx906.

Keep each BF16 RNE boundary in FP32 using integer bit operations. Triton 3.8
on this DTK stack miscompiles some intermediate 16-bit values when they are
broadcast after reduction. Final output stores retain the declared BF16 dtype.
No external compute kernels or handwritten Triton/HIP code are used here.
"""
import ninetoothed.language as ntl
from ninetoothed.language import libdevice

def swiglu_apply(up, gate, out):
    value = gate
    value = value.to(ntl.float32)
    round_value_0 = (value / (1.0 + libdevice.exp((-value).to(ntl.float64)).to(ntl.float32))).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    activated = round_value_0
    out = up * activated

def rms_apply(x, weight, eps, out):
    value = x
    value = value.to(ntl.float32)
    round_value_0 = (value * value).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    squared = round_value_0.to(ntl.float32)
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.0)
    round_value_1 = (ntl.sum(squared, 0) / x.source.shape[-1]).to(ntl.float32)
    round_bits_1 = round_value_1.to(ntl.uint32, bitcast=True)
    round_rne_1 = round_bits_1 + 32767 + (round_bits_1 >> 16 & 1) & 4294901760
    round_rne_1 = ntl.where(round_bits_1 & 2147483647 > 2139095040, round_bits_1 & 4294901760 | 4194304, round_rne_1)
    round_value_1 = round_rne_1.to(ntl.float32, bitcast=True)
    mean = round_value_1
    epsilon = eps
    round_value_2 = epsilon.to(ntl.float32).to(ntl.float32)
    round_bits_2 = round_value_2.to(ntl.uint32, bitcast=True)
    round_rne_2 = round_bits_2 + 32767 + (round_bits_2 >> 16 & 1) & 4294901760
    round_rne_2 = ntl.where(round_bits_2 & 2147483647 > 2139095040, round_bits_2 & 4294901760 | 4194304, round_rne_2)
    round_value_2 = round_rne_2.to(ntl.float32, bitcast=True)
    epsilon = round_value_2
    round_value_3 = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(ntl.float32)
    round_bits_3 = round_value_3.to(ntl.uint32, bitcast=True)
    round_rne_3 = round_bits_3 + 32767 + (round_bits_3 >> 16 & 1) & 4294901760
    round_rne_3 = ntl.where(round_bits_3 & 2147483647 > 2139095040, round_bits_3 & 4294901760 | 4194304, round_rne_3)
    round_value_3 = round_rne_3.to(ntl.float32, bitcast=True)
    variance = round_value_3
    round_value_4 = ntl.rsqrt(variance.to(ntl.float32)).to(ntl.float32)
    round_bits_4 = round_value_4.to(ntl.uint32, bitcast=True)
    round_rne_4 = round_bits_4 + 32767 + (round_bits_4 >> 16 & 1) & 4294901760
    round_rne_4 = ntl.where(round_bits_4 & 2147483647 > 2139095040, round_bits_4 & 4294901760 | 4194304, round_rne_4)
    round_value_4 = round_rne_4.to(ntl.float32, bitcast=True)
    reciprocal = round_value_4
    round_value_5 = (value * reciprocal.to(ntl.float32)).to(ntl.float32)
    round_bits_5 = round_value_5.to(ntl.uint32, bitcast=True)
    round_rne_5 = round_bits_5 + 32767 + (round_bits_5 >> 16 & 1) & 4294901760
    round_rne_5 = ntl.where(round_bits_5 & 2147483647 > 2139095040, round_bits_5 & 4294901760 | 4194304, round_rne_5)
    round_value_5 = round_rne_5.to(ntl.float32, bitcast=True)
    normalized = round_value_5
    out = normalized * weight

def rms_modulate_apply(x, weight, eps, scale, shift, out):
    value = x
    value = value.to(ntl.float32)
    round_value_0 = (value * value).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    squared = round_value_0.to(ntl.float32)
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.0)
    round_value_1 = (ntl.sum(squared, 0) / x.source.shape[-1]).to(ntl.float32)
    round_bits_1 = round_value_1.to(ntl.uint32, bitcast=True)
    round_rne_1 = round_bits_1 + 32767 + (round_bits_1 >> 16 & 1) & 4294901760
    round_rne_1 = ntl.where(round_bits_1 & 2147483647 > 2139095040, round_bits_1 & 4294901760 | 4194304, round_rne_1)
    round_value_1 = round_rne_1.to(ntl.float32, bitcast=True)
    mean = round_value_1
    epsilon = eps
    round_value_2 = epsilon.to(ntl.float32).to(ntl.float32)
    round_bits_2 = round_value_2.to(ntl.uint32, bitcast=True)
    round_rne_2 = round_bits_2 + 32767 + (round_bits_2 >> 16 & 1) & 4294901760
    round_rne_2 = ntl.where(round_bits_2 & 2147483647 > 2139095040, round_bits_2 & 4294901760 | 4194304, round_rne_2)
    round_value_2 = round_rne_2.to(ntl.float32, bitcast=True)
    epsilon = round_value_2
    round_value_3 = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(ntl.float32)
    round_bits_3 = round_value_3.to(ntl.uint32, bitcast=True)
    round_rne_3 = round_bits_3 + 32767 + (round_bits_3 >> 16 & 1) & 4294901760
    round_rne_3 = ntl.where(round_bits_3 & 2147483647 > 2139095040, round_bits_3 & 4294901760 | 4194304, round_rne_3)
    round_value_3 = round_rne_3.to(ntl.float32, bitcast=True)
    variance = round_value_3
    round_value_4 = ntl.rsqrt(variance.to(ntl.float32)).to(ntl.float32)
    round_bits_4 = round_value_4.to(ntl.uint32, bitcast=True)
    round_rne_4 = round_bits_4 + 32767 + (round_bits_4 >> 16 & 1) & 4294901760
    round_rne_4 = ntl.where(round_bits_4 & 2147483647 > 2139095040, round_bits_4 & 4294901760 | 4194304, round_rne_4)
    round_value_4 = round_rne_4.to(ntl.float32, bitcast=True)
    reciprocal = round_value_4
    round_value_5 = (value * reciprocal.to(ntl.float32)).to(ntl.float32)
    round_bits_5 = round_value_5.to(ntl.uint32, bitcast=True)
    round_rne_5 = round_bits_5 + 32767 + (round_bits_5 >> 16 & 1) & 4294901760
    round_rne_5 = ntl.where(round_bits_5 & 2147483647 > 2139095040, round_bits_5 & 4294901760 | 4194304, round_rne_5)
    round_value_5 = round_rne_5.to(ntl.float32, bitcast=True)
    normalized = round_value_5
    round_value_6 = (normalized * weight).to(ntl.float32)
    round_bits_6 = round_value_6.to(ntl.uint32, bitcast=True)
    round_rne_6 = round_bits_6 + 32767 + (round_bits_6 >> 16 & 1) & 4294901760
    round_rne_6 = ntl.where(round_bits_6 & 2147483647 > 2139095040, round_bits_6 & 4294901760 | 4194304, round_rne_6)
    round_value_6 = round_rne_6.to(ntl.float32, bitcast=True)
    weighted = round_value_6
    round_value_7 = (1.0 + scale).to(ntl.float32)
    round_bits_7 = round_value_7.to(ntl.uint32, bitcast=True)
    round_rne_7 = round_bits_7 + 32767 + (round_bits_7 >> 16 & 1) & 4294901760
    round_rne_7 = ntl.where(round_bits_7 & 2147483647 > 2139095040, round_bits_7 & 4294901760 | 4194304, round_rne_7)
    round_value_7 = round_rne_7.to(ntl.float32, bitcast=True)
    factor = round_value_7
    round_value_8 = (weighted * factor).to(ntl.float32)
    round_bits_8 = round_value_8.to(ntl.uint32, bitcast=True)
    round_rne_8 = round_bits_8 + 32767 + (round_bits_8 >> 16 & 1) & 4294901760
    round_rne_8 = ntl.where(round_bits_8 & 2147483647 > 2139095040, round_bits_8 & 4294901760 | 4194304, round_rne_8)
    round_value_8 = round_rne_8.to(ntl.float32, bitcast=True)
    product = round_value_8
    out = product + shift

def gated_residual_apply(x, gate, y, out):
    round_value_0 = (gate * y).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    product = round_value_0
    out = x + product

def gated_residual_rms_modulate_apply(x, gate, y, weight, eps, scale, shift, residual, out):
    round_value_0 = (gate * y).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    product = round_value_0
    round_value_1 = (x + product).to(ntl.float32)
    round_bits_1 = round_value_1.to(ntl.uint32, bitcast=True)
    round_rne_1 = round_bits_1 + 32767 + (round_bits_1 >> 16 & 1) & 4294901760
    round_rne_1 = ntl.where(round_bits_1 & 2147483647 > 2139095040, round_bits_1 & 4294901760 | 4194304, round_rne_1)
    round_value_1 = round_rne_1.to(ntl.float32, bitcast=True)
    summed = round_value_1
    residual = summed
    value = summed.to(ntl.float32)
    round_value_2 = (value * value).to(ntl.float32)
    round_bits_2 = round_value_2.to(ntl.uint32, bitcast=True)
    round_rne_2 = round_bits_2 + 32767 + (round_bits_2 >> 16 & 1) & 4294901760
    round_rne_2 = ntl.where(round_bits_2 & 2147483647 > 2139095040, round_bits_2 & 4294901760 | 4194304, round_rne_2)
    round_value_2 = round_rne_2.to(ntl.float32, bitcast=True)
    squared = round_value_2.to(ntl.float32)
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.0)
    round_value_3 = (ntl.sum(squared, 0) / x.source.shape[-1]).to(ntl.float32)
    round_bits_3 = round_value_3.to(ntl.uint32, bitcast=True)
    round_rne_3 = round_bits_3 + 32767 + (round_bits_3 >> 16 & 1) & 4294901760
    round_rne_3 = ntl.where(round_bits_3 & 2147483647 > 2139095040, round_bits_3 & 4294901760 | 4194304, round_rne_3)
    round_value_3 = round_rne_3.to(ntl.float32, bitcast=True)
    mean = round_value_3
    epsilon = eps
    round_value_4 = epsilon.to(ntl.float32).to(ntl.float32)
    round_bits_4 = round_value_4.to(ntl.uint32, bitcast=True)
    round_rne_4 = round_bits_4 + 32767 + (round_bits_4 >> 16 & 1) & 4294901760
    round_rne_4 = ntl.where(round_bits_4 & 2147483647 > 2139095040, round_bits_4 & 4294901760 | 4194304, round_rne_4)
    round_value_4 = round_rne_4.to(ntl.float32, bitcast=True)
    epsilon = round_value_4
    round_value_5 = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(ntl.float32)
    round_bits_5 = round_value_5.to(ntl.uint32, bitcast=True)
    round_rne_5 = round_bits_5 + 32767 + (round_bits_5 >> 16 & 1) & 4294901760
    round_rne_5 = ntl.where(round_bits_5 & 2147483647 > 2139095040, round_bits_5 & 4294901760 | 4194304, round_rne_5)
    round_value_5 = round_rne_5.to(ntl.float32, bitcast=True)
    variance = round_value_5
    round_value_6 = ntl.rsqrt(variance.to(ntl.float32)).to(ntl.float32)
    round_bits_6 = round_value_6.to(ntl.uint32, bitcast=True)
    round_rne_6 = round_bits_6 + 32767 + (round_bits_6 >> 16 & 1) & 4294901760
    round_rne_6 = ntl.where(round_bits_6 & 2147483647 > 2139095040, round_bits_6 & 4294901760 | 4194304, round_rne_6)
    round_value_6 = round_rne_6.to(ntl.float32, bitcast=True)
    reciprocal = round_value_6
    round_value_7 = (value * reciprocal.to(ntl.float32)).to(ntl.float32)
    round_bits_7 = round_value_7.to(ntl.uint32, bitcast=True)
    round_rne_7 = round_bits_7 + 32767 + (round_bits_7 >> 16 & 1) & 4294901760
    round_rne_7 = ntl.where(round_bits_7 & 2147483647 > 2139095040, round_bits_7 & 4294901760 | 4194304, round_rne_7)
    round_value_7 = round_rne_7.to(ntl.float32, bitcast=True)
    normalized = round_value_7
    round_value_8 = (normalized * weight).to(ntl.float32)
    round_bits_8 = round_value_8.to(ntl.uint32, bitcast=True)
    round_rne_8 = round_bits_8 + 32767 + (round_bits_8 >> 16 & 1) & 4294901760
    round_rne_8 = ntl.where(round_bits_8 & 2147483647 > 2139095040, round_bits_8 & 4294901760 | 4194304, round_rne_8)
    round_value_8 = round_rne_8.to(ntl.float32, bitcast=True)
    weighted = round_value_8
    round_value_9 = (1.0 + scale).to(ntl.float32)
    round_bits_9 = round_value_9.to(ntl.uint32, bitcast=True)
    round_rne_9 = round_bits_9 + 32767 + (round_bits_9 >> 16 & 1) & 4294901760
    round_rne_9 = ntl.where(round_bits_9 & 2147483647 > 2139095040, round_bits_9 & 4294901760 | 4194304, round_rne_9)
    round_value_9 = round_rne_9.to(ntl.float32, bitcast=True)
    factor = round_value_9
    round_value_10 = (weighted * factor).to(ntl.float32)
    round_bits_10 = round_value_10.to(ntl.uint32, bitcast=True)
    round_rne_10 = round_bits_10 + 32767 + (round_bits_10 >> 16 & 1) & 4294901760
    round_rne_10 = ntl.where(round_bits_10 & 2147483647 > 2139095040, round_bits_10 & 4294901760 | 4194304, round_rne_10)
    round_value_10 = round_rne_10.to(ntl.float32, bitcast=True)
    modulated = round_value_10
    out = modulated + shift

def rope_apply(x0, x1, passthrough, cos, sin, y0, y1, ypass):
    a = x0
    b = x1
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    round_value_0 = (a * c).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    ac = round_value_0.to(ntl.float32)
    round_value_1 = (-b * s).to(ntl.float32)
    round_bits_1 = round_value_1.to(ntl.uint32, bitcast=True)
    round_rne_1 = round_bits_1 + 32767 + (round_bits_1 >> 16 & 1) & 4294901760
    round_rne_1 = ntl.where(round_bits_1 & 2147483647 > 2139095040, round_bits_1 & 4294901760 | 4194304, round_rne_1)
    round_value_1 = round_rne_1.to(ntl.float32, bitcast=True)
    bs = round_value_1.to(ntl.float32)
    round_value_2 = (b * c).to(ntl.float32)
    round_bits_2 = round_value_2.to(ntl.uint32, bitcast=True)
    round_rne_2 = round_bits_2 + 32767 + (round_bits_2 >> 16 & 1) & 4294901760
    round_rne_2 = ntl.where(round_bits_2 & 2147483647 > 2139095040, round_bits_2 & 4294901760 | 4194304, round_rne_2)
    round_value_2 = round_rne_2.to(ntl.float32, bitcast=True)
    bc = round_value_2.to(ntl.float32)
    round_value_3 = (a * s).to(ntl.float32)
    round_bits_3 = round_value_3.to(ntl.uint32, bitcast=True)
    round_rne_3 = round_bits_3 + 32767 + (round_bits_3 >> 16 & 1) & 4294901760
    round_rne_3 = ntl.where(round_bits_3 & 2147483647 > 2139095040, round_bits_3 & 4294901760 | 4194304, round_rne_3)
    round_value_3 = round_rne_3.to(ntl.float32, bitcast=True)
    ass = round_value_3.to(ntl.float32)
    y0 = ac + bs
    y1 = bc + ass
    ypass = passthrough

def qk_norm_rope_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    round_value_0 = (a * a).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    sa = round_value_0.to(ntl.float32)
    round_value_1 = (b * b).to(ntl.float32)
    round_bits_1 = round_value_1.to(ntl.uint32, bitcast=True)
    round_rne_1 = round_bits_1 + 32767 + (round_bits_1 >> 16 & 1) & 4294901760
    round_rne_1 = ntl.where(round_bits_1 & 2147483647 > 2139095040, round_bits_1 & 4294901760 | 4194304, round_rne_1)
    round_value_1 = round_rne_1.to(ntl.float32, bitcast=True)
    sb = round_value_1.to(ntl.float32)
    round_value_2 = (p * p).to(ntl.float32)
    round_bits_2 = round_value_2.to(ntl.uint32, bitcast=True)
    round_rne_2 = round_bits_2 + 32767 + (round_bits_2 >> 16 & 1) & 4294901760
    round_rne_2 = ntl.where(round_bits_2 & 2147483647 > 2139095040, round_bits_2 & 4294901760 | 4194304, round_rne_2)
    round_value_2 = round_rne_2.to(ntl.float32, bitcast=True)
    sp = round_value_2.to(ntl.float32)
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.0)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.0)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.0)
    round_value_3 = ((ntl.sum(sa, 1) + ntl.sum(sb, 1) + ntl.sum(sp, 1)) / 128).to(ntl.float32)
    round_bits_3 = round_value_3.to(ntl.uint32, bitcast=True)
    round_rne_3 = round_bits_3 + 32767 + (round_bits_3 >> 16 & 1) & 4294901760
    round_rne_3 = ntl.where(round_bits_3 & 2147483647 > 2139095040, round_bits_3 & 4294901760 | 4194304, round_rne_3)
    round_value_3 = round_rne_3.to(ntl.float32, bitcast=True)
    mean = round_value_3
    epsilon = eps
    round_value_4 = epsilon.to(ntl.float32).to(ntl.float32)
    round_bits_4 = round_value_4.to(ntl.uint32, bitcast=True)
    round_rne_4 = round_bits_4 + 32767 + (round_bits_4 >> 16 & 1) & 4294901760
    round_rne_4 = ntl.where(round_bits_4 & 2147483647 > 2139095040, round_bits_4 & 4294901760 | 4194304, round_rne_4)
    round_value_4 = round_rne_4.to(ntl.float32, bitcast=True)
    epsilon = round_value_4
    round_value_5 = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(ntl.float32)
    round_bits_5 = round_value_5.to(ntl.uint32, bitcast=True)
    round_rne_5 = round_bits_5 + 32767 + (round_bits_5 >> 16 & 1) & 4294901760
    round_rne_5 = ntl.where(round_bits_5 & 2147483647 > 2139095040, round_bits_5 & 4294901760 | 4194304, round_rne_5)
    round_value_5 = round_rne_5.to(ntl.float32, bitcast=True)
    variance = round_value_5
    round_value_6 = ntl.rsqrt(variance.to(ntl.float32)).to(ntl.float32)
    round_bits_6 = round_value_6.to(ntl.uint32, bitcast=True)
    round_rne_6 = round_bits_6 + 32767 + (round_bits_6 >> 16 & 1) & 4294901760
    round_rne_6 = ntl.where(round_bits_6 & 2147483647 > 2139095040, round_bits_6 & 4294901760 | 4194304, round_rne_6)
    round_value_6 = round_rne_6.to(ntl.float32, bitcast=True)
    reciprocal = round_value_6.to(ntl.float32)
    round_value_7 = (a * reciprocal[:, None]).to(ntl.float32)
    round_bits_7 = round_value_7.to(ntl.uint32, bitcast=True)
    round_rne_7 = round_bits_7 + 32767 + (round_bits_7 >> 16 & 1) & 4294901760
    round_rne_7 = ntl.where(round_bits_7 & 2147483647 > 2139095040, round_bits_7 & 4294901760 | 4194304, round_rne_7)
    round_value_7 = round_rne_7.to(ntl.float32, bitcast=True)
    round_value_8 = (round_value_7 * w0).to(ntl.float32)
    round_bits_8 = round_value_8.to(ntl.uint32, bitcast=True)
    round_rne_8 = round_bits_8 + 32767 + (round_bits_8 >> 16 & 1) & 4294901760
    round_rne_8 = ntl.where(round_bits_8 & 2147483647 > 2139095040, round_bits_8 & 4294901760 | 4194304, round_rne_8)
    round_value_8 = round_rne_8.to(ntl.float32, bitcast=True)
    na = round_value_8.to(ntl.float32)
    round_value_9 = (b * reciprocal[:, None]).to(ntl.float32)
    round_bits_9 = round_value_9.to(ntl.uint32, bitcast=True)
    round_rne_9 = round_bits_9 + 32767 + (round_bits_9 >> 16 & 1) & 4294901760
    round_rne_9 = ntl.where(round_bits_9 & 2147483647 > 2139095040, round_bits_9 & 4294901760 | 4194304, round_rne_9)
    round_value_9 = round_rne_9.to(ntl.float32, bitcast=True)
    round_value_10 = (round_value_9 * w1).to(ntl.float32)
    round_bits_10 = round_value_10.to(ntl.uint32, bitcast=True)
    round_rne_10 = round_bits_10 + 32767 + (round_bits_10 >> 16 & 1) & 4294901760
    round_rne_10 = ntl.where(round_bits_10 & 2147483647 > 2139095040, round_bits_10 & 4294901760 | 4194304, round_rne_10)
    round_value_10 = round_rne_10.to(ntl.float32, bitcast=True)
    nb = round_value_10.to(ntl.float32)
    round_value_11 = (p * reciprocal[:, None]).to(ntl.float32)
    round_bits_11 = round_value_11.to(ntl.uint32, bitcast=True)
    round_rne_11 = round_bits_11 + 32767 + (round_bits_11 >> 16 & 1) & 4294901760
    round_rne_11 = ntl.where(round_bits_11 & 2147483647 > 2139095040, round_bits_11 & 4294901760 | 4194304, round_rne_11)
    round_value_11 = round_rne_11.to(ntl.float32, bitcast=True)
    round_value_12 = (round_value_11 * wp).to(ntl.float32)
    round_bits_12 = round_value_12.to(ntl.uint32, bitcast=True)
    round_rne_12 = round_bits_12 + 32767 + (round_bits_12 >> 16 & 1) & 4294901760
    round_rne_12 = ntl.where(round_bits_12 & 2147483647 > 2139095040, round_bits_12 & 4294901760 | 4194304, round_rne_12)
    round_value_12 = round_rne_12.to(ntl.float32, bitcast=True)
    npass = round_value_12
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    round_value_13 = (na * c).to(ntl.float32)
    round_bits_13 = round_value_13.to(ntl.uint32, bitcast=True)
    round_rne_13 = round_bits_13 + 32767 + (round_bits_13 >> 16 & 1) & 4294901760
    round_rne_13 = ntl.where(round_bits_13 & 2147483647 > 2139095040, round_bits_13 & 4294901760 | 4194304, round_rne_13)
    round_value_13 = round_rne_13.to(ntl.float32, bitcast=True)
    ac = round_value_13.to(ntl.float32)
    round_value_14 = (-nb * s).to(ntl.float32)
    round_bits_14 = round_value_14.to(ntl.uint32, bitcast=True)
    round_rne_14 = round_bits_14 + 32767 + (round_bits_14 >> 16 & 1) & 4294901760
    round_rne_14 = ntl.where(round_bits_14 & 2147483647 > 2139095040, round_bits_14 & 4294901760 | 4194304, round_rne_14)
    round_value_14 = round_rne_14.to(ntl.float32, bitcast=True)
    bs = round_value_14.to(ntl.float32)
    round_value_15 = (nb * c).to(ntl.float32)
    round_bits_15 = round_value_15.to(ntl.uint32, bitcast=True)
    round_rne_15 = round_bits_15 + 32767 + (round_bits_15 >> 16 & 1) & 4294901760
    round_rne_15 = ntl.where(round_bits_15 & 2147483647 > 2139095040, round_bits_15 & 4294901760 | 4194304, round_rne_15)
    round_value_15 = round_rne_15.to(ntl.float32, bitcast=True)
    bc = round_value_15.to(ntl.float32)
    round_value_16 = (na * s).to(ntl.float32)
    round_bits_16 = round_value_16.to(ntl.uint32, bitcast=True)
    round_rne_16 = round_bits_16 + 32767 + (round_bits_16 >> 16 & 1) & 4294901760
    round_rne_16 = ntl.where(round_bits_16 & 2147483647 > 2139095040, round_bits_16 & 4294901760 | 4194304, round_rne_16)
    round_value_16 = round_rne_16.to(ntl.float32, bitcast=True)
    ass = round_value_16.to(ntl.float32)
    y0 = ac + bs
    y1 = bc + ass
    yp = npass

def qk_norm_rope_ranks_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    round_value_0 = (a * a).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    sa = round_value_0.to(ntl.float32)
    round_value_1 = (b * b).to(ntl.float32)
    round_bits_1 = round_value_1.to(ntl.uint32, bitcast=True)
    round_rne_1 = round_bits_1 + 32767 + (round_bits_1 >> 16 & 1) & 4294901760
    round_rne_1 = ntl.where(round_bits_1 & 2147483647 > 2139095040, round_bits_1 & 4294901760 | 4194304, round_rne_1)
    round_value_1 = round_rne_1.to(ntl.float32, bitcast=True)
    sb = round_value_1.to(ntl.float32)
    round_value_2 = (p * p).to(ntl.float32)
    round_bits_2 = round_value_2.to(ntl.uint32, bitcast=True)
    round_rne_2 = round_bits_2 + 32767 + (round_bits_2 >> 16 & 1) & 4294901760
    round_rne_2 = ntl.where(round_bits_2 & 2147483647 > 2139095040, round_bits_2 & 4294901760 | 4194304, round_rne_2)
    round_value_2 = round_rne_2.to(ntl.float32, bitcast=True)
    sp = round_value_2.to(ntl.float32)
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.0)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.0)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.0)
    round_value_3 = ((ntl.sum(sa, 2) + ntl.sum(sb, 2) + ntl.sum(sp, 2)) / 128).to(ntl.float32)
    round_bits_3 = round_value_3.to(ntl.uint32, bitcast=True)
    round_rne_3 = round_bits_3 + 32767 + (round_bits_3 >> 16 & 1) & 4294901760
    round_rne_3 = ntl.where(round_bits_3 & 2147483647 > 2139095040, round_bits_3 & 4294901760 | 4194304, round_rne_3)
    round_value_3 = round_rne_3.to(ntl.float32, bitcast=True)
    mean = round_value_3
    epsilon = eps
    round_value_4 = epsilon.to(ntl.float32).to(ntl.float32)
    round_bits_4 = round_value_4.to(ntl.uint32, bitcast=True)
    round_rne_4 = round_bits_4 + 32767 + (round_bits_4 >> 16 & 1) & 4294901760
    round_rne_4 = ntl.where(round_bits_4 & 2147483647 > 2139095040, round_bits_4 & 4294901760 | 4194304, round_rne_4)
    round_value_4 = round_rne_4.to(ntl.float32, bitcast=True)
    epsilon = round_value_4
    round_value_5 = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(ntl.float32)
    round_bits_5 = round_value_5.to(ntl.uint32, bitcast=True)
    round_rne_5 = round_bits_5 + 32767 + (round_bits_5 >> 16 & 1) & 4294901760
    round_rne_5 = ntl.where(round_bits_5 & 2147483647 > 2139095040, round_bits_5 & 4294901760 | 4194304, round_rne_5)
    round_value_5 = round_rne_5.to(ntl.float32, bitcast=True)
    variance = round_value_5
    round_value_6 = ntl.rsqrt(variance.to(ntl.float32)).to(ntl.float32)
    round_bits_6 = round_value_6.to(ntl.uint32, bitcast=True)
    round_rne_6 = round_bits_6 + 32767 + (round_bits_6 >> 16 & 1) & 4294901760
    round_rne_6 = ntl.where(round_bits_6 & 2147483647 > 2139095040, round_bits_6 & 4294901760 | 4194304, round_rne_6)
    round_value_6 = round_rne_6.to(ntl.float32, bitcast=True)
    reciprocal = round_value_6.to(ntl.float32)
    round_value_7 = (a * reciprocal[:, :, None]).to(ntl.float32)
    round_bits_7 = round_value_7.to(ntl.uint32, bitcast=True)
    round_rne_7 = round_bits_7 + 32767 + (round_bits_7 >> 16 & 1) & 4294901760
    round_rne_7 = ntl.where(round_bits_7 & 2147483647 > 2139095040, round_bits_7 & 4294901760 | 4194304, round_rne_7)
    round_value_7 = round_rne_7.to(ntl.float32, bitcast=True)
    round_value_8 = (round_value_7 * w0).to(ntl.float32)
    round_bits_8 = round_value_8.to(ntl.uint32, bitcast=True)
    round_rne_8 = round_bits_8 + 32767 + (round_bits_8 >> 16 & 1) & 4294901760
    round_rne_8 = ntl.where(round_bits_8 & 2147483647 > 2139095040, round_bits_8 & 4294901760 | 4194304, round_rne_8)
    round_value_8 = round_rne_8.to(ntl.float32, bitcast=True)
    na = round_value_8.to(ntl.float32)
    round_value_9 = (b * reciprocal[:, :, None]).to(ntl.float32)
    round_bits_9 = round_value_9.to(ntl.uint32, bitcast=True)
    round_rne_9 = round_bits_9 + 32767 + (round_bits_9 >> 16 & 1) & 4294901760
    round_rne_9 = ntl.where(round_bits_9 & 2147483647 > 2139095040, round_bits_9 & 4294901760 | 4194304, round_rne_9)
    round_value_9 = round_rne_9.to(ntl.float32, bitcast=True)
    round_value_10 = (round_value_9 * w1).to(ntl.float32)
    round_bits_10 = round_value_10.to(ntl.uint32, bitcast=True)
    round_rne_10 = round_bits_10 + 32767 + (round_bits_10 >> 16 & 1) & 4294901760
    round_rne_10 = ntl.where(round_bits_10 & 2147483647 > 2139095040, round_bits_10 & 4294901760 | 4194304, round_rne_10)
    round_value_10 = round_rne_10.to(ntl.float32, bitcast=True)
    nb = round_value_10.to(ntl.float32)
    round_value_11 = (p * reciprocal[:, :, None]).to(ntl.float32)
    round_bits_11 = round_value_11.to(ntl.uint32, bitcast=True)
    round_rne_11 = round_bits_11 + 32767 + (round_bits_11 >> 16 & 1) & 4294901760
    round_rne_11 = ntl.where(round_bits_11 & 2147483647 > 2139095040, round_bits_11 & 4294901760 | 4194304, round_rne_11)
    round_value_11 = round_rne_11.to(ntl.float32, bitcast=True)
    round_value_12 = (round_value_11 * wp).to(ntl.float32)
    round_bits_12 = round_value_12.to(ntl.uint32, bitcast=True)
    round_rne_12 = round_bits_12 + 32767 + (round_bits_12 >> 16 & 1) & 4294901760
    round_rne_12 = ntl.where(round_bits_12 & 2147483647 > 2139095040, round_bits_12 & 4294901760 | 4194304, round_rne_12)
    round_value_12 = round_rne_12.to(ntl.float32, bitcast=True)
    npass = round_value_12
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    round_value_13 = (na * c).to(ntl.float32)
    round_bits_13 = round_value_13.to(ntl.uint32, bitcast=True)
    round_rne_13 = round_bits_13 + 32767 + (round_bits_13 >> 16 & 1) & 4294901760
    round_rne_13 = ntl.where(round_bits_13 & 2147483647 > 2139095040, round_bits_13 & 4294901760 | 4194304, round_rne_13)
    round_value_13 = round_rne_13.to(ntl.float32, bitcast=True)
    ac = round_value_13.to(ntl.float32)
    round_value_14 = (-nb * s).to(ntl.float32)
    round_bits_14 = round_value_14.to(ntl.uint32, bitcast=True)
    round_rne_14 = round_bits_14 + 32767 + (round_bits_14 >> 16 & 1) & 4294901760
    round_rne_14 = ntl.where(round_bits_14 & 2147483647 > 2139095040, round_bits_14 & 4294901760 | 4194304, round_rne_14)
    round_value_14 = round_rne_14.to(ntl.float32, bitcast=True)
    bs = round_value_14.to(ntl.float32)
    round_value_15 = (nb * c).to(ntl.float32)
    round_bits_15 = round_value_15.to(ntl.uint32, bitcast=True)
    round_rne_15 = round_bits_15 + 32767 + (round_bits_15 >> 16 & 1) & 4294901760
    round_rne_15 = ntl.where(round_bits_15 & 2147483647 > 2139095040, round_bits_15 & 4294901760 | 4194304, round_rne_15)
    round_value_15 = round_rne_15.to(ntl.float32, bitcast=True)
    bc = round_value_15.to(ntl.float32)
    round_value_16 = (na * s).to(ntl.float32)
    round_bits_16 = round_value_16.to(ntl.uint32, bitcast=True)
    round_rne_16 = round_bits_16 + 32767 + (round_bits_16 >> 16 & 1) & 4294901760
    round_rne_16 = ntl.where(round_bits_16 & 2147483647 > 2139095040, round_bits_16 & 4294901760 | 4194304, round_rne_16)
    round_value_16 = round_rne_16.to(ntl.float32, bitcast=True)
    ass = round_value_16.to(ntl.float32)
    y0 = ac + bs
    y1 = bc + ass
    yp = npass

def qk_norm_rope_block_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    round_value_0 = (a * a).to(ntl.float32)
    round_bits_0 = round_value_0.to(ntl.uint32, bitcast=True)
    round_rne_0 = round_bits_0 + 32767 + (round_bits_0 >> 16 & 1) & 4294901760
    round_rne_0 = ntl.where(round_bits_0 & 2147483647 > 2139095040, round_bits_0 & 4294901760 | 4194304, round_rne_0)
    round_value_0 = round_rne_0.to(ntl.float32, bitcast=True)
    sa = round_value_0.to(ntl.float32)
    round_value_1 = (b * b).to(ntl.float32)
    round_bits_1 = round_value_1.to(ntl.uint32, bitcast=True)
    round_rne_1 = round_bits_1 + 32767 + (round_bits_1 >> 16 & 1) & 4294901760
    round_rne_1 = ntl.where(round_bits_1 & 2147483647 > 2139095040, round_bits_1 & 4294901760 | 4194304, round_rne_1)
    round_value_1 = round_rne_1.to(ntl.float32, bitcast=True)
    sb = round_value_1.to(ntl.float32)
    round_value_2 = (p * p).to(ntl.float32)
    round_bits_2 = round_value_2.to(ntl.uint32, bitcast=True)
    round_rne_2 = round_bits_2 + 32767 + (round_bits_2 >> 16 & 1) & 4294901760
    round_rne_2 = ntl.where(round_bits_2 & 2147483647 > 2139095040, round_bits_2 & 4294901760 | 4194304, round_rne_2)
    round_value_2 = round_rne_2.to(ntl.float32, bitcast=True)
    sp = round_value_2.to(ntl.float32)
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.0)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.0)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.0)
    round_value_3 = ((ntl.sum(sa, 2) + ntl.sum(sb, 2) + ntl.sum(sp, 2)) / 128).to(ntl.float32)
    round_bits_3 = round_value_3.to(ntl.uint32, bitcast=True)
    round_rne_3 = round_bits_3 + 32767 + (round_bits_3 >> 16 & 1) & 4294901760
    round_rne_3 = ntl.where(round_bits_3 & 2147483647 > 2139095040, round_bits_3 & 4294901760 | 4194304, round_rne_3)
    round_value_3 = round_rne_3.to(ntl.float32, bitcast=True)
    mean = round_value_3
    epsilon = eps
    round_value_4 = epsilon.to(ntl.float32).to(ntl.float32)
    round_bits_4 = round_value_4.to(ntl.uint32, bitcast=True)
    round_rne_4 = round_bits_4 + 32767 + (round_bits_4 >> 16 & 1) & 4294901760
    round_rne_4 = ntl.where(round_bits_4 & 2147483647 > 2139095040, round_bits_4 & 4294901760 | 4194304, round_rne_4)
    round_value_4 = round_rne_4.to(ntl.float32, bitcast=True)
    epsilon = round_value_4
    round_value_5 = (mean.to(ntl.float32) + epsilon.to(ntl.float32)).to(ntl.float32)
    round_bits_5 = round_value_5.to(ntl.uint32, bitcast=True)
    round_rne_5 = round_bits_5 + 32767 + (round_bits_5 >> 16 & 1) & 4294901760
    round_rne_5 = ntl.where(round_bits_5 & 2147483647 > 2139095040, round_bits_5 & 4294901760 | 4194304, round_rne_5)
    round_value_5 = round_rne_5.to(ntl.float32, bitcast=True)
    variance = round_value_5
    round_value_6 = ntl.rsqrt(variance.to(ntl.float32)).to(ntl.float32)
    round_bits_6 = round_value_6.to(ntl.uint32, bitcast=True)
    round_rne_6 = round_bits_6 + 32767 + (round_bits_6 >> 16 & 1) & 4294901760
    round_rne_6 = ntl.where(round_bits_6 & 2147483647 > 2139095040, round_bits_6 & 4294901760 | 4194304, round_rne_6)
    round_value_6 = round_rne_6.to(ntl.float32, bitcast=True)
    reciprocal = round_value_6.to(ntl.float32)
    round_value_7 = (a * reciprocal[:, :, None]).to(ntl.float32)
    round_bits_7 = round_value_7.to(ntl.uint32, bitcast=True)
    round_rne_7 = round_bits_7 + 32767 + (round_bits_7 >> 16 & 1) & 4294901760
    round_rne_7 = ntl.where(round_bits_7 & 2147483647 > 2139095040, round_bits_7 & 4294901760 | 4194304, round_rne_7)
    round_value_7 = round_rne_7.to(ntl.float32, bitcast=True)
    round_value_8 = (round_value_7 * w0).to(ntl.float32)
    round_bits_8 = round_value_8.to(ntl.uint32, bitcast=True)
    round_rne_8 = round_bits_8 + 32767 + (round_bits_8 >> 16 & 1) & 4294901760
    round_rne_8 = ntl.where(round_bits_8 & 2147483647 > 2139095040, round_bits_8 & 4294901760 | 4194304, round_rne_8)
    round_value_8 = round_rne_8.to(ntl.float32, bitcast=True)
    na = round_value_8.to(ntl.float32)
    round_value_9 = (b * reciprocal[:, :, None]).to(ntl.float32)
    round_bits_9 = round_value_9.to(ntl.uint32, bitcast=True)
    round_rne_9 = round_bits_9 + 32767 + (round_bits_9 >> 16 & 1) & 4294901760
    round_rne_9 = ntl.where(round_bits_9 & 2147483647 > 2139095040, round_bits_9 & 4294901760 | 4194304, round_rne_9)
    round_value_9 = round_rne_9.to(ntl.float32, bitcast=True)
    round_value_10 = (round_value_9 * w1).to(ntl.float32)
    round_bits_10 = round_value_10.to(ntl.uint32, bitcast=True)
    round_rne_10 = round_bits_10 + 32767 + (round_bits_10 >> 16 & 1) & 4294901760
    round_rne_10 = ntl.where(round_bits_10 & 2147483647 > 2139095040, round_bits_10 & 4294901760 | 4194304, round_rne_10)
    round_value_10 = round_rne_10.to(ntl.float32, bitcast=True)
    nb = round_value_10.to(ntl.float32)
    round_value_11 = (p * reciprocal[:, :, None]).to(ntl.float32)
    round_bits_11 = round_value_11.to(ntl.uint32, bitcast=True)
    round_rne_11 = round_bits_11 + 32767 + (round_bits_11 >> 16 & 1) & 4294901760
    round_rne_11 = ntl.where(round_bits_11 & 2147483647 > 2139095040, round_bits_11 & 4294901760 | 4194304, round_rne_11)
    round_value_11 = round_rne_11.to(ntl.float32, bitcast=True)
    round_value_12 = (round_value_11 * wp).to(ntl.float32)
    round_bits_12 = round_value_12.to(ntl.uint32, bitcast=True)
    round_rne_12 = round_bits_12 + 32767 + (round_bits_12 >> 16 & 1) & 4294901760
    round_rne_12 = ntl.where(round_bits_12 & 2147483647 > 2139095040, round_bits_12 & 4294901760 | 4194304, round_rne_12)
    round_value_12 = round_rne_12.to(ntl.float32, bitcast=True)
    npass = round_value_12
    c = cos
    s = sin
    c = c.to(ntl.float32)[:, None, :]
    s = s.to(ntl.float32)[:, None, :]
    round_value_13 = (na * c).to(ntl.float32)
    round_bits_13 = round_value_13.to(ntl.uint32, bitcast=True)
    round_rne_13 = round_bits_13 + 32767 + (round_bits_13 >> 16 & 1) & 4294901760
    round_rne_13 = ntl.where(round_bits_13 & 2147483647 > 2139095040, round_bits_13 & 4294901760 | 4194304, round_rne_13)
    round_value_13 = round_rne_13.to(ntl.float32, bitcast=True)
    ac = round_value_13.to(ntl.float32)
    round_value_14 = (-nb * s).to(ntl.float32)
    round_bits_14 = round_value_14.to(ntl.uint32, bitcast=True)
    round_rne_14 = round_bits_14 + 32767 + (round_bits_14 >> 16 & 1) & 4294901760
    round_rne_14 = ntl.where(round_bits_14 & 2147483647 > 2139095040, round_bits_14 & 4294901760 | 4194304, round_rne_14)
    round_value_14 = round_rne_14.to(ntl.float32, bitcast=True)
    bs = round_value_14.to(ntl.float32)
    round_value_15 = (nb * c).to(ntl.float32)
    round_bits_15 = round_value_15.to(ntl.uint32, bitcast=True)
    round_rne_15 = round_bits_15 + 32767 + (round_bits_15 >> 16 & 1) & 4294901760
    round_rne_15 = ntl.where(round_bits_15 & 2147483647 > 2139095040, round_bits_15 & 4294901760 | 4194304, round_rne_15)
    round_value_15 = round_rne_15.to(ntl.float32, bitcast=True)
    bc = round_value_15.to(ntl.float32)
    round_value_16 = (na * s).to(ntl.float32)
    round_bits_16 = round_value_16.to(ntl.uint32, bitcast=True)
    round_rne_16 = round_bits_16 + 32767 + (round_bits_16 >> 16 & 1) & 4294901760
    round_rne_16 = ntl.where(round_bits_16 & 2147483647 > 2139095040, round_bits_16 & 4294901760 | 4194304, round_rne_16)
    round_value_16 = round_rne_16.to(ntl.float32, bitcast=True)
    ass = round_value_16.to(ntl.float32)
    y0 = ac + bs
    y1 = bc + ass
    yp = npass

APPLICATIONS = {fn.__name__: fn for fn in (gated_residual_apply, gated_residual_rms_modulate_apply, qk_norm_rope_apply, qk_norm_rope_block_apply, qk_norm_rope_ranks_apply, rms_apply, rms_modulate_apply, rope_apply, swiglu_apply)}
