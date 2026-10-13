"""Compare captured FL2VA image/text conditioning with full Transformers Qwen."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3FL2VATextEncoderStep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dump", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = torch.load(args.input_dump, weights_only=True)
    assert capture["diagnostic_only"]
    actual = capture["inputs"]["encoder_hidden_states"]
    processor = AutoProcessor.from_pretrained(str(args.model_path / "processor"), local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model_path / "text_encoder"),
                dtype=torch.bfloat16, device_map="balanced_low_0", max_memory={0:"54GiB",1:"54GiB"},
                low_cpu_mem_usage=True, local_files_only=True).eval().requires_grad_(False)
    components = SimpleNamespace(text_encoder=encoder, processor=processor, tokenizer=processor.tokenizer,
            text_encoder_layer=50, text_tag=1, video_tag=0, _execution_device=torch.device("cuda:0"))
    # Exact square-canvas preprocessing used by the official first-keyframe block.
    keyframe = Image.open(args.image).convert("RGB").resize((768,768), Image.Resampling.LANCZOS)
    state = PipelineState(values={"prompt":args.text, "keyframes":[keyframe]})
    with torch.inference_mode():
        _,state = MiniMaxH3FL2VATextEncoderStep()(components,state)
    reference = state.prompt_embeds.cpu()
    assert reference.shape == actual.shape
    tags = state.text_token_tags.cpu()
    assert torch.equal(tags, capture["inputs"]["token_tags"][capture["inputs"]["text_indices"]])
    report = {"diagnostic_only":True, "num_layers":encoder.config.text_config.num_hidden_layers,
              "device_map":encoder.hf_device_map, "shape":list(reference.shape), "groups":{}}
    for label,mask in (("all",torch.ones_like(tags,dtype=torch.bool)),("vision",tags==0),("text",tags==1)):
        a,b = actual[:,mask].float(), reference[:,mask].float()
        delta = a-b
        report["groups"][label] = {"relative_l2":float(delta.norm()/b.norm()), "max_abs":float(delta.abs().max()),
                                   "actual_rms":float(a.square().mean().sqrt()),
                                   "reference_rms":float(b.square().mean().sqrt())}
    torch.save({"diagnostic_only":True, "reference_conditioning":reference}, args.output_dir/"reference-conditioning.pt")
    report["pass"] = report["groups"]["all"]["relative_l2"] < 1e-2
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
