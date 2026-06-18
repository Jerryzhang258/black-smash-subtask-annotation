"""
Qwen2.5-VL environment smoke test.

Builds a synthetic test image, loads the model, runs ONE image->text inference,
and prints env info + VRAM usage. Use this to confirm the install (see
docs/INSTALL_QWEN.md) works before running the real Stage-1 annotator.

Run:
  python test_qwen_vl.py --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ
  python test_qwen_vl.py --model Qwen/Qwen2.5-VL-3B-Instruct
"""
import argparse, time, tempfile, os, sys


def make_test_image(path):
    """A simple, unambiguous scene the VLM should be able to describe."""
    from PIL import Image, ImageDraw
    W, H = 640, 480
    im = Image.new("RGB", (W, H), (30, 30, 40))
    d = ImageDraw.Draw(im)
    # vertical gradient
    for y in range(H):
        d.line([(0, y), (W, y)], fill=(30, 30 + y // 6, 80))
    d.ellipse([220, 150, 420, 350], fill=(220, 60, 60))           # red circle
    d.rectangle([60, 60, 200, 200], fill=(60, 200, 90))           # green square
    d.polygon([(500, 380), (440, 460), (560, 460)], fill=(240, 210, 0))  # yellow triangle
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local dir or HF/ModelScope id")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Describe this image in one sentence. "
                                        "What colored shapes do you see?")
    args = ap.parse_args()

    import torch
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    print(f"[env] torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | cap {cap}")
    if not torch.cuda.is_available():
        print("!! CUDA not available — fix torch install (see docs step 3) before continuing.")
        sys.exit(1)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        process_vision_info = None

    img_path = os.path.join(tempfile.gettempdir(), "qwen_vl_test.png")
    make_test_image(img_path)

    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(args.model)
    torch.cuda.synchronize()
    print(f"[load] model loaded in {time.time()-t0:.1f}s, "
          f"VRAM allocated {torch.cuda.memory_allocated()/1e9:.1f} GB")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        from PIL import Image
        image_inputs, video_inputs = [Image.open(img_path)], None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    t1 = time.time()
    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    print(f"[infer] {time.time()-t1:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("[infer] output:", out.strip())
    print("OK ✅")


if __name__ == "__main__":
    main()
