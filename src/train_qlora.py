import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


LEGACY_CLOSED_PROMPT = (
    "Answer the medical visual question using only one word: yes or no.\n"
    "Question: {question}\n"
    "Answer:"
)
LEGACY_OPEN_PROMPT = (
    "You are a pathology visual question answering model.\n"
    "Answer the question with the exact medical entity, disease name, structure name, process name, "
    "location, or short phrase.\n"
    "Do not explain.\n"
    "Question: {question}\n"
    "Answer:"
)
OPEN_FOCUS_CLOSED_PROMPT = (
    "You are a pathology visual question answering assistant. "
    "Answer the question based on the image. "
    "For yes/no questions, answer only Yes or No.\n"
    "Question: {question}\n"
    "Answer:"
)
OPEN_FOCUS_OPEN_PROMPT = (
    "You are a pathology visual question answering assistant. "
    "Answer the question based on the image. "
    "For open-ended questions, answer with a short medical term or phrase. "
    "Do not explain. Do not output a full sentence.\n"
    "Question: {question}\n"
    "Answer:"
)


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA instruction tuning for MedGemma on PathVQA.")
    parser.add_argument("--model_id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--train_jsonl", default="data/pathvqa_sft/train.jsonl")
    parser.add_argument("--val_jsonl", default="data/pathvqa_sft/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/medgemma_pathvqa_qlora")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--prompt_variant", choices=("legacy", "open_focus"), default="legacy")
    parser.add_argument("--train_log_output", default=None)
    parser.add_argument("--max_steps", type=int, default=-1, help="Use 10 for a sanity-check run.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--dataloader_smoke_samples", type=int, default=0)
    parser.add_argument("--loss_on_answer_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed_loss_weight", type=float, default=1.0)
    parser.add_argument("--debug_loss_tokens", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path, max_samples=None):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def question_type_for_record(record):
    return record.get("question_type") or record.get("answer_type")


def prompt_for_record(record, prompt_variant="legacy"):
    question_type = question_type_for_record(record)
    if prompt_variant == "open_focus":
        template = OPEN_FOCUS_CLOSED_PROMPT if question_type == "closed" else OPEN_FOCUS_OPEN_PROMPT
    else:
        template = LEGACY_CLOSED_PROMPT if question_type == "closed" else LEGACY_OPEN_PROMPT
    return template.format(question=record["question"])


class PathVQASFTDataset:
    def __init__(self, jsonl_path, max_samples=None):
        self.records = read_jsonl(jsonl_path, max_samples=max_samples)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class MedGemmaSFTCollator:
    def __init__(
        self,
        processor,
        max_seq_len,
        loss_on_answer_only=True,
        closed_loss_weight=1.0,
        debug_loss_tokens=True,
        prompt_variant="legacy",
    ):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.loss_on_answer_only = loss_on_answer_only
        self.closed_loss_weight = closed_loss_weight
        self.debug_loss_tokens = debug_loss_tokens
        self.prompt_variant = prompt_variant
        self._debug_printed = False
        self.tokenizer = getattr(processor, "tokenizer", processor)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode_one(self, record):
        import torch

        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_for_record(record, self.prompt_variant)},
                    ],
                }
            ]
            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": record["answer"]}],
                }
            ]
            prompt_inputs = self.processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"text_kwargs": {"truncation": True, "max_length": self.max_seq_len}},
            )
            full_inputs = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"text_kwargs": {"truncation": True, "max_length": self.max_seq_len}},
            )

        def as_single_tensor(value):
            if hasattr(value, "squeeze"):
                return value.squeeze(0)
            if isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], list):
                    value = value[0]
                return torch.tensor(value)
            return value

        encoded = {key: as_single_tensor(value) for key, value in full_inputs.items()}
        labels = encoded["input_ids"].clone()
        if self.loss_on_answer_only:
            prompt_ids = as_single_tensor(prompt_inputs["input_ids"])
            prompt_len = min(prompt_ids.shape[-1], labels.shape[-1])
            labels[:prompt_len] = -100
        else:
            prompt_len = 0
        encoded["labels"] = labels
        loss_mask = labels.ne(-100).to(dtype=encoded["input_ids"].dtype)
        question_type = question_type_for_record(record)
        if question_type == "closed" and self.closed_loss_weight != 1.0:
            loss_mask = loss_mask * self.closed_loss_weight
        encoded["loss_weight_mask"] = loss_mask
        encoded["_answer_type"] = question_type
        if self.debug_loss_tokens and not self._debug_printed:
            trained_ids = encoded["input_ids"][labels.ne(-100)]
            trained_text = self.tokenizer.decode(trained_ids, skip_special_tokens=True).strip()
            print(
                "DEBUG loss tokens "
                f"sample_id={record['sample_id']} answer_type={question_type} "
                f"prompt_len={prompt_len} trained_text={trained_text!r}"
            )
            self._debug_printed = True
        return encoded

    def __call__(self, records):
        import torch

        features = [self._encode_one(record) for record in records]
        batch = {}
        pad_token_id = self.tokenizer.pad_token_id
        all_keys = set().union(*(feature.keys() for feature in features))
        for key in all_keys:
            if key.startswith("_"):
                continue
            values = [feature[key] for feature in features if key in feature]
            if len(values) != len(features):
                continue
            if key in {"input_ids", "attention_mask", "labels", "loss_weight_mask"}:
                max_len = max(value.shape[-1] for value in values)
                padded = []
                for value in values:
                    pad_len = max_len - value.shape[-1]
                    if key == "labels":
                        pad_value = -100
                    elif key == "input_ids":
                        pad_value = pad_token_id
                    else:
                        pad_value = 0
                    padded.append(torch.nn.functional.pad(value, (0, pad_len), value=pad_value))
                batch[key] = torch.stack(padded)
            else:
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values
        return batch


def dataset_closed_counts(dataset, name):
    answers = Counter(record["answer"] for record in dataset.records if question_type_for_record(record) == "closed")
    print(
        f"{name} closed distribution: "
        f"yes={answers.get('yes', 0)}, no={answers.get('no', 0)}, total={sum(answers.values())}"
    )


class WeightedLossTrainer:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch

        loss_weight_mask = inputs.pop("loss_weight_mask", None)
        if loss_weight_mask is None:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weight_mask[..., 1:].to(device=shift_logits.device, dtype=shift_logits.dtype)
        valid = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid, 0)
        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            safe_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        weighted = per_token_loss * shift_weights * valid.to(dtype=shift_logits.dtype)
        denom = (shift_weights * valid.to(dtype=shift_logits.dtype)).sum().clamp_min(1.0)
        loss = weighted.sum() / denom
        return (loss, outputs) if return_outputs else loss


def smoke_test_dataloader(dataset, collator, sample_count, batch_size):
    from torch.utils.data import DataLoader, Subset

    count = min(sample_count, len(dataset))
    loader = DataLoader(Subset(dataset, range(count)), batch_size=batch_size, collate_fn=collator)
    seen = 0
    for batch in loader:
        batch_size_actual = batch["input_ids"].shape[0]
        seen += batch_size_actual
        print(
            f"dataloader batch: batch_size={batch_size_actual}, "
            f"seq_len={batch['input_ids'].shape[1]}, keys={sorted(batch.keys())}"
        )
    print(f"Dataloader smoke test read {seen} samples.")


def find_lora_target_modules(model):
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query",
        "key",
        "value",
        "dense",
    }
    found = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates:
            found.add(leaf)
    if found:
        return sorted(found)
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def freeze_vision_encoder(model):
    frozen = 0
    for name, param in model.named_parameters():
        lowered = name.lower()
        if any(token in lowered for token in ("vision", "image", "visual", "siglip")):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def main():
    args = parse_args()

    try:
        import torch
        from transformers import AutoProcessor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency {exc.name!r}. Install: python -m pip install torch transformers accelerate pillow"
        ) from exc

    processor = AutoProcessor.from_pretrained(args.model_id)
    train_dataset = PathVQASFTDataset(args.train_jsonl, max_samples=args.max_train_samples)
    val_dataset = PathVQASFTDataset(args.val_jsonl, max_samples=args.max_val_samples)
    collator = MedGemmaSFTCollator(
        processor=processor,
        max_seq_len=args.max_seq_len,
        loss_on_answer_only=args.loss_on_answer_only,
        closed_loss_weight=args.closed_loss_weight,
        debug_loss_tokens=args.debug_loss_tokens,
        prompt_variant=args.prompt_variant,
    )
    dataset_closed_counts(train_dataset, "train")
    dataset_closed_counts(val_dataset, "val")

    if args.dataloader_smoke_samples:
        smoke_test_dataloader(train_dataset, collator, args.dataloader_smoke_samples, args.batch_size)
        if args.max_steps == 0:
            return

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, Trainer, TrainingArguments
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency {exc.name!r}. Install: python -m pip install peft bitsandbytes"
        ) from exc

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    frozen_vision_params = freeze_vision_encoder(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_lora_target_modules(model),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"Frozen vision/image parameters: {frozen_vision_params}")
    print(f"Using compute dtype: {compute_dtype}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
    )
    class PathVQATrainer(WeightedLossTrainer, Trainer):
        pass

    trainer = PathVQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config["base_model_init"] = args.model_id
    run_config["lora"] = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": sorted(lora_config.target_modules),
    }
    (output_dir / "training_args.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.train_log_output:
        log_path = Path(args.train_log_output)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# QLoRA v2 e2 Open Focus Train Log",
            "",
            f"- model_id: `{args.model_id}`",
            f"- train_jsonl: `{args.train_jsonl}`",
            f"- val_jsonl: `{args.val_jsonl}`",
            f"- output_dir: `{args.output_dir}`",
            f"- epochs: {args.epochs}",
            f"- learning_rate: {args.lr}",
            f"- prompt_variant: `{args.prompt_variant}`",
            f"- loss_on_answer_only: {args.loss_on_answer_only}",
            f"- closed_loss_weight: {args.closed_loss_weight}",
            f"- final_global_step: {trainer.state.global_step}",
            f"- final_epoch: {trainer.state.epoch}",
            "",
            "## Expected Saved Artifacts",
            "",
            f"- `{output_dir / 'trainer_state.json'}`",
            f"- `{output_dir / 'adapter_config.json'}`",
            f"- `{output_dir / 'adapter_model.safetensors'}`",
            f"- `{output_dir / 'training_args.json'}`",
        ]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
