import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from datasets import Dataset, DatasetDict


SPLIT_MAP = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


def load_codetask_split(file_path: Path, task_name: str, split_name: str) -> Dataset:
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    definition = payload.get("Definition", [])
    positive_examples = payload.get("Positive Examples", [])
    negative_examples = payload.get("Negative Examples", [])

    rows: List[Dict[str, Any]] = []
    for item in payload.get("Instances", []):
        outputs = item.get("output", [])
        rows.append(
            {
                "task": task_name,
                "split": split_name,
                "id": item.get("id", ""),
                "input": item.get("input", ""),
                "output": outputs[0] if outputs else "",
                "outputs": outputs,
                "definition": definition,
                "positive_examples": json.dumps(positive_examples, ensure_ascii=False),
                "negative_examples": json.dumps(negative_examples, ensure_ascii=False),
            }
        )

    return Dataset.from_list(rows)


def build_task_dataset(task_dir: Path) -> DatasetDict:
    ds_dict: Dict[str, Dataset] = {}
    task_name = task_dir.name

    for raw_split, hf_split in SPLIT_MAP.items():
        split_file = task_dir / f"{raw_split}.json"
        if split_file.exists():
            ds_dict[hf_split] = load_codetask_split(split_file, task_name, hf_split)

    if not ds_dict:
        raise ValueError(f"No split files found in {task_dir}")

    return DatasetDict(ds_dict)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload CodeTask JSON datasets to Hugging Face Hub"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/scratch/projects/punim1928/HUST/east/CodeGR/Dense/Online-CL-LLMs/CODETASK_Benchmark",
        help="Path containing task folders (BFP, CodeSearchNet, ...)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hub dataset repo id, e.g. yourname/codetask",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token. If omitted, existing login will be used.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update a private dataset repository",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Upload only one task folder name, e.g. BFP",
    )
    parser.add_argument(
        "--push-merged",
        action="store_true",
        help="Also push merged train/validation/test as config named 'all'",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    task_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if args.task:
        task_dirs = [p for p in task_dirs if p.name == args.task]

    if not task_dirs:
        raise ValueError("No task folders matched your selection")

    merged_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}

    for task_dir in task_dirs:
        task_name = task_dir.name
        task_ds = build_task_dataset(task_dir)
        task_ds.push_to_hub(
            repo_id=args.repo_id,
            config_name=task_name,
            token=args.token,
            private=args.private,
        )
        print(f"Uploaded config: {task_name}")

        if args.push_merged:
            for split in ["train", "validation", "test"]:
                if split in task_ds:
                    merged_rows[split].extend(task_ds[split].to_list())

    if args.push_merged:
        merged_ds = DatasetDict(
            {
                split: Dataset.from_list(rows)
                for split, rows in merged_rows.items()
                if rows
            }
        )
        merged_ds.push_to_hub(
            repo_id=args.repo_id,
            config_name="all",
            token=args.token,
            private=args.private,
        )
        print("Uploaded config: all")


if __name__ == "__main__":
    main()
