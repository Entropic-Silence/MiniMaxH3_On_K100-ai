"""Exact full, noncausal H3 attention expressed solely in NineToothed."""
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(q, k, v, scale, o, block_m=128, block_n=32, block_d=128):
    def arrange_qo(x):
        a = x.tile((1, 1, block_m, block_d))
        a.dtype = a.dtype.squeeze((0, 1))
        return a

    def arrange_kv(x):
        a = x.tile((1, 1, block_n, block_d)).tile((1, 1, -1, -1)).expand(
            (-1, -1, q_arr.shape[-2], -1))
        a.dtype = a.dtype.squeeze((0, 1, 3))
        a.dtype.dtype = a.dtype.dtype.squeeze((0, 1))
        return a

    q_arr = arrange_qo(q)
    return q_arr, arrange_kv(k), arrange_kv(v), scale, arrange_qo(o)


def application(q, k, v, scale, o):
    # Bind the Q tile once: NineToothed inlines every reference to an input as
    # a load, so using `q` inside the loop would re-read it for each K/V block.
    query = q
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    denominator = ntl.full((q.shape[-2],), 0, dtype=ntl.float32)
    maximum = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)
    for j in range(k.shape[0]):
        logits = ntl.dot(query, ntl.trans(k[j])) * (scale.to(ntl.float32) * 1.4426950408889634)
        logits = ntl.where(k[j].offsets(-2) < k.source.shape[-2], logits, float("-inf"))
        new_maximum = ntl.maximum(maximum, ntl.max(logits, 1))
        weights = ntl.exp2(logits - new_maximum[:, None])
        correction = ntl.exp2(maximum - new_maximum)
        acc = acc * correction[:, None] + ntl.dot(weights.to(v[j].dtype), v[j])
        denominator = denominator * correction + ntl.sum(weights, 1)
        maximum = new_maximum
    o = acc / denominator[:, None]  # noqa: F841


def application_dotacc(q, k, v, scale, o):
    query = q
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    denominator = ntl.full((q.shape[-2],), 0, dtype=ntl.float32)
    maximum = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)
    for j in range(k.shape[0]):
        logits = ntl.dot(query, ntl.trans(k[j])) * (scale.to(ntl.float32) * 1.4426950408889634)
        logits = ntl.where(k[j].offsets(-2) < k.source.shape[-2], logits, float("-inf"))
        new_maximum = ntl.maximum(maximum, ntl.max(logits, 1))
        weights = ntl.exp2(logits - new_maximum[:, None])
        correction = ntl.exp2(maximum - new_maximum)
        acc = ntl.dot(weights.to(v[j].dtype), v[j], acc * correction[:, None])
        denominator = denominator * correction + ntl.sum(weights, 1)
        maximum = new_maximum
    o = acc / denominator[:, None]  # noqa: F841


def application_split(q, k, v, scale, o):
    # Same arithmetic as `application`; only the key-bound mask is confined to
    # the final K/V block. Earlier blocks are always complete.
    query = q
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    denominator = ntl.full((q.shape[-2],), 0, dtype=ntl.float32)
    maximum = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)
    for j in range(k.shape[0] - 1):
        logits = ntl.dot(query, ntl.trans(k[j])) * (scale.to(ntl.float32) * 1.4426950408889634)
        new_maximum = ntl.maximum(maximum, ntl.max(logits, 1))
        weights = ntl.exp2(logits - new_maximum[:, None])
        correction = ntl.exp2(maximum - new_maximum)
        acc = acc * correction[:, None] + ntl.dot(weights.to(v[j].dtype), v[j])
        denominator = denominator * correction + ntl.sum(weights, 1)
        maximum = new_maximum
    last = k.shape[0] - 1
    logits = ntl.dot(query, ntl.trans(k[last])) * (scale.to(ntl.float32) * 1.4426950408889634)
    logits = ntl.where(k[last].offsets(-2) < k.source.shape[-2], logits, float("-inf"))
    new_maximum = ntl.maximum(maximum, ntl.max(logits, 1))
    weights = ntl.exp2(logits - new_maximum[:, None])
    correction = ntl.exp2(maximum - new_maximum)
    acc = acc * correction[:, None] + ntl.dot(weights.to(v[last].dtype), v[last])
    denominator = denominator * correction + ntl.sum(weights, 1)
    o = acc / denominator[:, None]  # noqa: F841


def application_prescale(q, k, v, scale, o):
    scaled_q = (scale.to(ntl.float32) * 1.4426950408889634 * q).to(q.dtype)
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    denominator = ntl.full((q.shape[-2],), 0, dtype=ntl.float32)
    maximum = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)
    for j in range(k.shape[0]):
        logits = ntl.dot(scaled_q, ntl.trans(k[j]))
        logits = ntl.where(k[j].offsets(-2) < k.source.shape[-2], logits, float("-inf"))
        new_maximum = ntl.maximum(maximum, ntl.max(logits, 1))
        weights = ntl.exp2(logits - new_maximum[:, None])
        correction = ntl.exp2(maximum - new_maximum)
        acc = acc * correction[:, None] + ntl.dot(weights.to(v[j].dtype), v[j])
        denominator = denominator * correction + ntl.sum(weights, 1)
        maximum = new_maximum
    o = acc / denominator[:, None]  # noqa: F841


def make_attention(block_m=128, block_n=32, num_warps=4, num_stages=1,
                   caller="torch", output_dir=None, archs=None, backend_options=None,
                   kernel_name="h3_full_attention", numerics="fp32_qk", head_dim=128,
                   dtype=ninetoothed.bfloat16):
    # B,H,S are dynamic; D is the model's actual head dimension, not a padded approximation.
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    tensors = tuple(Tensor(shape=(None, None, None, head_dim), dtype=dtype)
                    for _ in range(3)) + (Tensor(0, dtype=ninetoothed.float64),
                    Tensor(shape=(None, None, None, head_dim), dtype=dtype))
    applications = {"fp32_qk": application, "bf16_prescale": application_prescale,
                    "fp32_qk_dotacc": application_dotacc, "fp32_qk_split": application_split}
    if numerics not in applications:
        raise ValueError(f"Unknown attention numerics: {numerics}")
    return ninetoothed.make(functools.partial(arrangement, block_m=block_m, block_n=block_n,
                                             block_d=1 << (head_dim - 1).bit_length()),
        applications[numerics], tensors, caller=caller, kernel_name=kernel_name, output_dir=output_dir,
        num_warps=num_warps, num_stages=num_stages, archs=archs, backend_options=backend_options)
