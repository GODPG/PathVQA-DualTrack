import os
import re
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image


CLOSED_PROMPT = (
    "Answer the pathology visual question using only one word: yes or no.\n"
    "Do not explain.\n"
    "Question: {question}\n"
    "Answer:"
)

OPEN_PROMPT = (
    "You are a pathology visual question answering model.\n"
    "Answer the question with the exact medical entity, disease name, structure name, "
    "process name, location, number, color, or short phrase that best completes the question.\n"
    "Do not describe the image.\n"
    "Do not repeat the question.\n"
    "Do not explain.\n"
    "Do not list multiple possible answers.\n"
    "Output exactly one short phrase in one line.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def is_closed(answer_type):
    return str(answer_type).strip().lower() in {"yes/no", "closed", "binary"}


def get_prompt(row):
    if is_closed(row.get("answer_type", "")):
        return CLOSED_PROMPT.format(question=row["question"])
    return OPEN_PROMPT.format(question=row["question"])


def clean_prediction(text, closed=False):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^the answer is\s+", "", text, flags=re.I)
    text = re.sub(r"^it is\s+", "", text, flags=re.I)
    text = text.strip().lower()

    if closed:
        if re.search(r"\byes\b", text):
            return "yes"
        if re.search(r"\bno\b", text):
            return "no"

    for stop in ["\n", ".", ";", "!", "?"]:
        if stop in text:
            text = text.split(stop)[0].strip()

    for prefix in ["a ", "an ", "the "]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    return text.strip(" ,:;\"'")


def resolve_image_path(path, image_root=None):
    path = str(path)
    if os.path.exists(path):
        return path
    if image_root:
        candidate = Path(image_root) / Path(path).name
        if candidate.exists():
            return str(candidate)
    return path


def load_existing(output_path):
    if not os.path.exists(output_path):
        return [], set()
    df = pd.read_csv(output_path)
    done = set(df["sample_id"].astype(str))
    return df.to_dict("records"), done


def load_qwen_model(model_path):
    from transformers import AutoConfig, AutoProcessor

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    elif model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
    else:
        from transformers import AutoModelForImageTextToText as ModelCls

    model = ModelCls.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return processor, model


def ask_qwen(processor, model, image_path, prompt, max_new_tokens):
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated)
    ]

    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def load_internvl_model(model_path):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().cuda()
    return tokenizer, model


def ask_internvl(tokenizer, model, image_path, prompt, max_new_tokens):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    question = "<image>\n" + prompt
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    with torch.inference_mode():
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    return str(response).strip()


def load_llava_med_model(model_path, llava_repo):
    sys.path.insert(0, llava_repo)

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model, image_processor


def ask_llava_med(tokenizer, model, image_processor, image_path, prompt, max_new_tokens, conv_mode):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    new_tokens = output_ids[:, input_ids.shape[1]:]
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    if not text:
        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["qwen", "internvl", "llava_med"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--closed_max_new_tokens", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--llava_repo", default="/root/autodl-tmp/LLaVA-Med")
    parser.add_argument("--conv_mode", default="mistral_instruct")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit)

    print("Model:", args.model_name)
    print("Backend:", args.backend)
    print("Samples:", len(df))

    if args.backend == "qwen":
        processor, model = load_qwen_model(args.model_path)
    elif args.backend == "internvl":
        tokenizer, model = load_internvl_model(args.model_path)
    else:
        tokenizer, model, image_processor = load_llava_med_model(args.model_path, args.llava_repo)

    rows, done = load_existing(args.output) if args.resume else ([], set())

    step = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["sample_id"])
        if sample_id in done:
            continue

        closed = is_closed(row.get("answer_type", ""))
        prompt = get_prompt(row)
        image_path = resolve_image_path(row["image_path"], args.image_root)
        max_new_tokens = args.closed_max_new_tokens if closed else args.max_new_tokens

        if not os.path.exists(image_path):
            raw = ""
            pred = ""
            error = f"Image not found: {image_path}"
        else:
            try:
                if args.backend == "qwen":
                    raw = ask_qwen(processor, model, image_path, prompt, max_new_tokens)
                elif args.backend == "internvl":
                    raw = ask_internvl(tokenizer, model, image_path, prompt, max_new_tokens)
                else:
                    raw = ask_llava_med(
                        tokenizer, model, image_processor, image_path,
                        prompt, max_new_tokens, args.conv_mode
                    )
                pred = clean_prediction(raw, closed=closed)
                error = ""
            except Exception as e:
                raw = ""
                pred = ""
                error = str(e)

        rows.append({
            "model": args.model_name,
            "prompt_type": "fixed_type_aware",
            "sample_id": sample_id,
            "image_path": image_path,
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "answer_type": row.get("answer_type", ""),
            "question_type": row.get("question_type", ""),
            "prediction_raw": raw,
            "prediction": pred,
            "error": error,
        })

        done.add(sample_id)
        step += 1

        if step % args.save_every == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")

    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()