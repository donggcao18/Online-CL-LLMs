import os
import json
import hashlib 
import numpy as np
from typing import Any, Dict
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from task_info import TASK_SPECS, TASK_LIST
FOLDER_NAME = os.path.dirname(os.path.abspath(__file__))

def _to_string(value):
    if value is None:
        return ""
    return str(value)
    
def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")

def _load_split(repo_id: str, split: str) -> Dataset:
    ds = load_dataset(
    repo_id,
    split=split,
    revision="e85e1e6c871ee584381c4215af06b6b072cd8b02",
    token=_hf_token(),
    download_mode="reuse_cache_if_exists"
    )
    return ds


def _limit_dataset(dataset: Dataset, max_samples: int=-1, seed: int=0) -> Dataset:
    if max_samples == -1 or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def _load_training_dataset(language, max_train_samples, seed=0) -> Dataset:
    split_datasets = []
    for split in ["train_OSS_Instruct", "train_McEval_Instruct"]:
        dataset = _load_split("ankhanhtran02/CL4Code-executable-datasets", split)
        dataset = dataset.filter(
            lambda row: row["language"] == language and row["solution"] is not None
        )
        split_datasets.append(dataset)

    if len(split_datasets) != 2:
        raise ValueError(f"Expected to load 2 training splits, but got {len(split_datasets)}")
    train_dataset = concatenate_datasets(split_datasets)
    train_dataset = _limit_dataset(train_dataset, max_train_samples, seed)
    # if len(dataset) > 0:
    #     print("[train] Sample:")
    #     print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    return train_dataset

def _load_eval_dataset(language, max_eval_samples, seed=0) -> Dataset:
    dataset = _load_split("ankhanhtran02/CL4Code-executable-datasets", "test_McEval")
    dataset = dataset.filter(
        lambda row: row["language"] == language and row["test"] is not None
    )
    dataset = _limit_dataset(dataset, max_eval_samples, seed)
    if len(dataset) == 0:
        raise ValueError(f"No evaluation samples found in split=test_McEval for language={language}.")
    # if len(dataset) > 0:
    #     print("[eval] Sample:")
    #     print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    return dataset


def create_executable_dataset(dataset_name, seed, num_train, num_eval, num_test):
    train_dataset = _load_training_dataset(dataset_name, num_train, seed)
    test_dataset = _load_eval_dataset(dataset_name, num_test, seed)
    eval_dataset = _load_eval_dataset(dataset_name, num_eval, seed)
    return train_dataset, eval_dataset, test_dataset


def convert_to_executable(
    split_seed=42,
    max_dev_samples=1000,
    max_test_samples=1000,
    max_train_samples=None,
):  
    if max_train_samples is None:
        max_train_samples = -1
    if max_dev_samples is None:
        max_dev_samples = -1
    if max_test_samples is None:
        max_test_samples = -1
    for task in TASK_LIST:
        try:
            save_dir = os.path.join(FOLDER_NAME, task)
            os.makedirs(save_dir, exist_ok=True)
            train_dataset, eval_dataset, test_dataset = create_executable_dataset(
                dataset_name=task,
                seed=split_seed,
                num_train=max_train_samples,
                num_eval=max_dev_samples,
                num_test=max_test_samples,
            )
            print(f"Loaded datasets for {task}: train={len(train_dataset)}, dev={len(eval_dataset)}, test={len(test_dataset)}")

            for dataset, split_name in zip([train_dataset, eval_dataset, test_dataset], ["train", "validation", "test"]):
                output_data = {
                    "Definition": [TASK_SPECS['definition'].format(language=task)],
                    "Positive Examples": [],
                    "Negative Examples": [],
                    "Instances": []
                }

                text_key = TASK_SPECS['text_key']
                label_key = TASK_SPECS['label_key']
                for example in dataset:
                    input_text = _to_string(example.get(text_key))
                    output_text = _to_string(example.get(label_key))

                    uid = hashlib.md5((task + "||" + input_text).encode("utf-8")).hexdigest()
                    instruction_input = f"{input_text}".strip()

                    output_data["Instances"].append({
                        "id": f"{task}-{uid}",
                        "input": instruction_input,
                        "output": [output_text]
                    })

                with open(os.path.join(save_dir, f"{split_name}.json"), "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️  Skipped {task}: {e}")


if __name__ == "__main__":
    np.random.seed(42)
    convert_to_executable(
        split_seed=42,
        max_dev_samples=-1,
        max_test_samples=-1,
        max_train_samples=-1
    )
