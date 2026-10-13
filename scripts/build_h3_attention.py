#!/usr/bin/env python3
"""Build relocatable H3 HIP AOT artifacts for InfiniCore."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ninetoothed/src"), str(ROOT / "tools")]
from h3_attention import make_attention
from h3_pointwise import (make_gated_residual, make_gated_residual_rms_modulate, make_qk_norm_rope,
                          make_rms, make_rms_modulate, make_rope, make_rope_cache, make_swiglu)
from ninetoothed.platform import current_target

parser = argparse.ArgumentParser()
parser.add_argument("--archs", help="comma-separated; defaults to actual device architecture")
parser.add_argument("--config", type=Path)
parser.add_argument("--output", type=Path, default=ROOT / "InfiniCore/build/h3_ninetoothed")
args = parser.parse_args()
config_path = args.config or ROOT / "configs" / current_target().arch / "attention.json"
config = json.loads(config_path.read_text()) if config_path.is_file() else {}
archs = args.archs.split(",") if args.archs else [current_target().arch]
make_attention(caller="hip", output_dir=args.output, archs=archs, **config)
make_attention(caller="hip", output_dir=args.output, archs=archs, head_dim=72,
               kernel_name="h3_vision_attention_72", **config)
make_swiglu(caller="hip", output_dir=args.output, archs=archs)
for width in (128, 5376):
    make_rms(caller="hip", output_dir=args.output, archs=archs, width=width)
make_rms_modulate(caller="hip", output_dir=args.output, archs=archs)
make_rope(caller="hip", output_dir=args.output, archs=archs)
make_gated_residual(caller="hip", output_dir=args.output, archs=archs)
make_gated_residual_rms_modulate(caller="hip", output_dir=args.output, archs=archs)
make_qk_norm_rope(caller="hip", output_dir=args.output, archs=archs)
make_rope_cache(caller="hip", output_dir=args.output, archs=archs)
print(json.dumps({"event": "built", "archs": archs, "output": str(args.output), "config": config}))
