import argparse
import os
import sys
from typing import Any, Dict, List

from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cl_collator import DataCollator  # noqa: E402


class _ConfigStub:
    def __init__(self, model_name: str):
        self._name_or_path = model_name


class _ModelStub:
    def __init__(self, model_name: str):
        self.config = _ConfigStub(model_name)

    def prepare_decoder_input_ids_from_labels(self, labels):
        return labels


def _preview(text: Any, limit: int = 140) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _analyze_split(split_ds, split_name: str, sample_count: int, scan_limit: int):
    total = len(split_ds)
    scan_n = min(total, scan_limit)
    empty_sentence = 0
    empty_label = 0
    unresolved_placeholder = 0

    for i in range(scan_n):
        ex = split_ds[i]
        inst = ex["Instance"]
        sentence = inst.get("sentence", "")
        label = inst.get("label", "")
        instruction = inst.get("instruction", "")

        if not str(sentence).strip():
            empty_sentence += 1
        if not str(label).strip():
            empty_label += 1
        if "{0}" in str(instruction):
            unresolved_placeholder += 1

    print(f"\n=== SPLIT: {split_name} ===")
    print(f"Total rows: {total}")
    print(f"Scanned rows: {scan_n}")
    print(f"Empty sentence rows: {empty_sentence}")
    print(f"Empty label rows: {empty_label}")
    print(f"Instruction templates containing {{0}}: {unresolved_placeholder}")

    show_n = min(total, sample_count)
    print(f"\nFirst {show_n} raw examples:")
    for i in range(show_n):
        ex = split_ds[i]
        inst = ex["Instance"]
        print(f"  [{i}] Dataset={ex['Dataset']} id={inst.get('id', '')}")
        print(f"      sentence(len={len(str(inst.get('sentence', '')))}): {_preview(inst.get('sentence', ''))}")
        print(f"      label(len={len(str(inst.get('label', '')))}): {_preview(inst.get('label', ''))}")
        print(f"      instruction(len={len(str(inst.get('instruction', '')))}): {_preview(inst.get('instruction', ''))}")


def _debug_collator(
    split_ds,
    split_name: str,
    tokenizer,
    model_name: str,
    max_source_length: int,
    max_target_length: int,
    add_instruction_replay: bool,
    sample_count: int,
):
    if len(split_ds) == 0:
        print(f"\n=== COLLATOR: {split_name} ===")
        print("Split is empty, skipping collator debug.")
        return

    collator = DataCollator(
        tokenizer=tokenizer,
        model=_ModelStub(model_name),
        padding="longest",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_instruction_replay=add_instruction_replay,
        text_only=True,
    )

    show_n = min(len(split_ds), sample_count)
    batch = [split_ds[i] for i in range(show_n)]

    print(f"\n=== COLLATOR: {split_name} ===")
    print(f"Rows in debug batch: {show_n}")
    print(f"add_instruction_replay={add_instruction_replay}")

    model_inputs = collator(batch)
    sources: List[str] = model_inputs["inputs"]
    labels: List[str] = model_inputs["labels"]

    for i, (ex, source, label) in enumerate(zip(batch, sources, labels)):
        inst = ex["Instance"]
        rendered_direct = collator.get_instruction(ex)
        unresolved = "{0}" in rendered_direct
        print(f"  [{i}] Dataset={ex['Dataset']} id={inst.get('id', '')}")
        print(f"      raw sentence: {_preview(inst.get('sentence', ''))}")
        print(f"      rendered get_instruction unresolved={{0}}? {unresolved}")
        print(f"      source(len={len(source)}): {_preview(source)}")
        print(f"      label(len={len(label)}): {_preview(label)}")


def main():
    parser = argparse.ArgumentParser(description="Debug dataset loading and DataCollator behavior for train/test splits.")
    parser.add_argument("--data_dir", required=True, help="Path passed to run script as --data_dir")
    parser.add_argument("--task_config_dir", required=True, help="Path passed to run script as --task_config_dir")
    parser.add_argument("--tokenizer_name_or_path", required=True, help="Tokenizer path/model id used in training")
    parser.add_argument("--model_name_hint", default="t5", help="Model name hint for collator routing, e.g., t5 or llama")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--sample_count", type=int, default=3, help="How many rows to print for raw and collated inspection")
    parser.add_argument("--scan_limit", type=int, default=5000, help="How many rows to scan for empty sentence/label statistics")
    parser.add_argument(
        "--disable_instruction_replay",
        action="store_true",
        help="Disable instruction formatting in collator to compare behavior",
    )
    args = parser.parse_args()

    dataset_script = os.path.join(REPO_ROOT, "src", "cl_dataset.py")
    download_config = DownloadConfig(local_files_only=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)

    print("Loading dataset through src/cl_dataset.py...")
    raw_datasets = load_dataset(
        dataset_script,
        data_dir=args.data_dir,
        task_config_dir=args.task_config_dir,
        download_config=download_config,
        max_num_instances_per_task=None,
        max_num_instances_per_eval_task=None,
        num_examples=0,
    )

    has_train = "train" in raw_datasets
    has_test = "test" in raw_datasets

    print("\nLoaded splits:", list(raw_datasets.keys()))
    if not has_train and not has_test:
        raise RuntimeError("No train/test split found in loaded dataset.")

    if has_train:
        _analyze_split(raw_datasets["train"], "train", args.sample_count, args.scan_limit)
        _debug_collator(
            raw_datasets["train"],
            "train",
            tokenizer,
            args.model_name_hint,
            args.max_source_length,
            args.max_target_length,
            not args.disable_instruction_replay,
            args.sample_count,
        )

    if has_test:
        _analyze_split(raw_datasets["test"], "test", args.sample_count, args.scan_limit)
        _debug_collator(
            raw_datasets["test"],
            "test",
            tokenizer,
            args.model_name_hint,
            args.max_source_length,
            args.max_target_length,
            not args.disable_instruction_replay,
            args.sample_count,
        )


if __name__ == "__main__":
    main()