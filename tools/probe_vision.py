"""Isolate complete Qwen vision features using real weights and FL2VA preprocessing."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3FL2VATextEncoderStep
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-conditioning", type=Path,
                        help="Same-case diagnostic full-Qwen output from probe_conditioner_image.py")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["H3_TEXT_ENCODER"] = "sequential"
    os.environ["H3_QWEN_TRACE_DIR"] = str(args.output_dir)
    processor = AutoProcessor.from_pretrained(str(args.model_path / "processor"), local_files_only=True)
    encoder = MiniMaxH3ConditionEncoder(args.model_path / "text_encoder", tp_device_ids=[0,1])
    captured = {}
    def save_inputs(module, positional, kwargs):
        captured.update({k:v.detach().cpu() for k,v in kwargs.items() if isinstance(v,torch.Tensor)})
    hook = encoder.model.register_forward_pre_hook(save_inputs, with_kwargs=True)
    components = SimpleNamespace(text_encoder=encoder, processor=processor, tokenizer=processor.tokenizer,
            text_encoder_layer=50, text_tag=1, video_tag=0, _execution_device=torch.device("cuda:0"))
    keyframe = Image.open(args.image).convert("RGB").resize((768,768), Image.Resampling.LANCZOS)
    state = PipelineState(values={"prompt":args.text,"keyframes":[keyframe]})
    with torch.inference_mode():
        _,state = MiniMaxH3FL2VATextEncoderStep()(components,state)
    hook.remove()
    torch.save({"diagnostic_only":True, "conditioning":state.prompt_embeds.cpu(),
                "inputs":captured}, args.output_dir / "native-conditioning.pt")
    encoder._residency.release()
    os.environ.pop("H3_QWEN_TRACE_DIR", None)
    # This is the complete released vision component, no truncated layers or
    # overridden attention backend; it is not an end-to-end legacy reference.
    config = AutoConfig.from_pretrained(str(args.model_path / "text_encoder"),local_files_only=True).vision_config
    vision = Qwen3VLVisionModel(config)
    weights = {}
    for file in sorted((args.model_path / "text_encoder").glob("*.safetensors")):
        with safe_open(file,framework="pt",device="cpu") as source:
            for key in source.keys():
                if key.startswith("model.visual."):
                    weights[key.removeprefix("model.visual.")] = source.get_tensor(key)
    vision.load_state_dict(weights, strict=True, assign=True)
    vision.to("cuda:0").eval().requires_grad_(False)
    layer_references = {}
    def save_stage(name):
        def hook(module, inputs, output):
            layer_references[name] = output[:64].detach().cpu()
        return hook
    vision.patch_embed.register_forward_hook(save_stage("patch"))
    def packed_hook(module, inputs):
        layer_references["packed"] = inputs[0][:64].detach().cpu()
    vision.blocks[0].register_forward_pre_hook(packed_hook)
    for i,block in enumerate(vision.blocks):
        block.register_forward_hook(save_stage(f"block{i}"))
    for name in ("norm1","attn","norm2","mlp"):
        getattr(vision.blocks[0],name).register_forward_hook(save_stage("block0."+name))
    with torch.inference_mode():
        result = vision(captured["pixel_values"].to("cuda:0", dtype=torch.bfloat16),
                        grid_thw=captured["image_grid_thw"].to("cuda:0"))
    references = {"pool":result.pooler_output.cpu()}
    references.update({f"deep{i}":value.cpu() for i,value in enumerate(result.deepstack_features)})
    report = {"diagnostic_only":True,"vision_config":config.to_dict(),
              "attention_implementation":vision.config._attn_implementation,
              "fp32_rope":os.environ.get("H3_QWEN_FP32_ROPE","0"),
              "ninetoothed_attention":os.environ.get("H3_QWEN_NINETOOTHED","0"),
              "image_grid_thw":captured["image_grid_thw"].tolist(),"stages":{}}
    for name, ref in references.items():
        actual = torch.frombuffer(bytearray((args.output_dir/f"image0.{name}.bin").read_bytes()),
                                  dtype=torch.bfloat16).reshape(ref.shape)
        delta = actual.float()-ref.float()
        report["stages"][name] = {"shape":list(ref.shape),"relative_l2":float(delta.norm()/ref.float().norm()),
                                  "max_abs":float(delta.abs().max()),"finite":bool(actual.isfinite().all())}
    report["layers"] = {}
    for name,ref in layer_references.items():
        file = args.output_dir/f"vision.{name}.bin"
        if file.exists():
            actual = torch.frombuffer(bytearray(file.read_bytes()),dtype=torch.bfloat16).reshape(ref.shape)
            report["layers"][name] = {"relative_l2":float((actual.float()-ref.float()).norm()/ref.float().norm()),
                                      "max_abs":float((actual.float()-ref.float()).abs().max())}
    if args.reference_conditioning:
        data = torch.load(args.reference_conditioning,weights_only=True)
        assert data["diagnostic_only"]
        ref,actual = data["reference_conditioning"].float(), state.prompt_embeds.cpu().float()
        assert actual.shape == ref.shape
        report["conditioning"] = {"reference":str(args.reference_conditioning),
                                   "relative_l2":float((actual-ref).norm()/ref.norm()),
                                   "max_abs":float((actual-ref).abs().max())}
    torch.save({"diagnostic_only":True,"features":references},args.output_dir/"reference-vision.pt")
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
