import os
import json
import hashlib 
import numpy as np
from typing import Any, Dict
from datasets import load_dataset, concatenate_datasets
from task_info import TASK_SPECS, HF_SPLIT_MAP, INSTRUCTION_POOL, TRAIN_ONLY_TASKS, TASK_LIST, INSTRUCTION_SPLIT_POLICY
FOLDER_NAME = os.path.dirname(os.path.abspath(__file__))


def _split_train_only(dataset, task, split, split_seed=42):
    sizes = TRAIN_ONLY_TASKS[task]
    test_size = sizes['test']
    val_size = sizes['val']

    tmp = dataset.train_test_split(test_size=test_size, seed=split_seed)
    test_ds = tmp['test']

    tmp2 = tmp['train'].train_test_split(test_size=val_size, seed=split_seed)
    train_ds = tmp2['train']
    val_ds = tmp2['test']

    mapping = {'train': train_ds, 'validation': val_ds, 'test': test_ds}
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}' for train-only task '{task}'")
    return mapping[split]


def _load_task_split(task, split_name, split_seed=42):
    spec = TASK_SPECS[task]

    if task == 'TheVault_Csharp':
        split_map = {
            'train': ['train/small'],
            'validation': ['validation'],
            'test': ['test'],
        }
        dataset_dict = load_dataset(
            spec['dataset_name'],
            languages=['c#'],
            split_set=split_map[split_name],
        )
        return concatenate_datasets(list(dataset_dict.values()))

    if task == 'KodCode':
        dataset = load_dataset(spec['dataset_name'], split='train')
        return _split_train_only(dataset, task, split_name, split_seed=split_seed)

    if task == 'RunBugRun':
        dataset = load_dataset(spec['dataset_name'], split='train')
        dataset = dataset.filter(lambda example: example['language'] == 'ruby')
        return _split_train_only(dataset, task, split_name, split_seed=split_seed)

    return load_dataset(spec['dataset_name'], split=HF_SPLIT_MAP[split_name])


def _to_string(value):
    if value is None:
        return ""
    return str(value)


def _get_candidate_instruction_pool(task_type: str, split_name: str):
    pool = INSTRUCTION_POOL.get(task_type, [])
    if not pool:
        raise ValueError(f"No instruction templates defined for task_type '{task_type}'")

    policy = INSTRUCTION_SPLIT_POLICY.get(split_name, INSTRUCTION_SPLIT_POLICY['train'])
    if policy['pool_scope'] == 'full':
        return pool

    if policy['pool_scope'] == 'head_fraction':
        fraction = float(policy.get('fraction', 0.75))
        if fraction <= 0:
            raise ValueError(f"Invalid fraction {fraction} for split '{split_name}'")
        head_size = max(1, int(len(pool) * fraction))
        return pool[:head_size]

    raise ValueError(f"Unknown pool_scope '{policy['pool_scope']}' for split '{split_name}'")


def _select_instruction_template(task_type: str, sample_key: str, split_name: str, split_seed: int) -> str:
    candidate_pool = _get_candidate_instruction_pool(task_type, split_name)
    random_key = f"{split_seed}::{split_name}::{sample_key}"
    idx = int(hashlib.md5(random_key.encode("utf-8")).hexdigest(), 16) % len(candidate_pool)
    return candidate_pool[idx]


def _render_instruction(task: str, raw_input: str, sample_key: str, split_name: str, split_seed: int) -> str:
    spec = TASK_SPECS[task]
    task_type = spec['task_type']
    template = _select_instruction_template(task_type, sample_key, split_name, split_seed)

    format_values: Dict[str, str] = {
        'language': spec.get('language', 'code'),
        'description': raw_input,
        'code': raw_input,
        'source_lang': spec.get('source_lang', spec.get('language', 'source language')),
        'target_lang': spec.get('target_lang', 'target language'),
    }
    return template.format(**format_values)


class CodeTaskPreprocessor:
    @staticmethod
    def _extract_first_paragraph(docstring: Any) -> str:
        if docstring is None:
            return ""
        if isinstance(docstring, (list, tuple)):
            s = " ".join(str(t) for t in docstring)
        else:
            s = str(docstring)
        s = s.replace("\n", "")
        s = " ".join(s.strip().split())
        return s


def convert_to_codetask(
    split_name="train",
    split_seed=42,
    max_dev_samples=1000,
    max_test_samples=1000,
    max_train_samples=None,
    use_instruction_pool=True,
):
    if split_name not in HF_SPLIT_MAP:
        raise ValueError(f"Unsupported split_name '{split_name}'. Use one of {list(HF_SPLIT_MAP.keys())}")

    for task in TASK_LIST:
        try:
            save_dir = os.path.join(FOLDER_NAME, task)
            os.makedirs(save_dir, exist_ok=True)
            dataset = _load_task_split(task, split_name, split_seed=split_seed)
            if split_name == 'train' and max_train_samples is not None:
                original_size = len(dataset)
                if original_size > max_train_samples:
                    dataset = dataset.select(range(max_train_samples))
                    print(f"[{task}::{split_name}] Truncated train set: {original_size} → {max_train_samples} samples")
                else:
                    print(f"[{task}::{split_name}] Train set size: {original_size} (max_train_samples={max_train_samples}, no truncation needed)")
            elif split_name in ('dev', 'validation') and max_dev_samples is not None:
                original_size = len(dataset)
                if original_size > max_dev_samples:
                    dataset = dataset.select(range(max_dev_samples))
                    print(f"[{task}::{split_name}] Truncated dev set: {original_size} → {max_dev_samples} samples")
                else:
                    print(f"[{task}::{split_name}] Dev set size: {original_size} (max_dev_samples={max_dev_samples}, no truncation needed)")
            elif split_name == 'test' and max_test_samples is not None:
                original_size = len(dataset)
                if original_size > max_test_samples:
                    dataset = dataset.select(range(max_test_samples))
                    print(f"[{task}::{split_name}] Truncated test set: {original_size} → {max_test_samples} samples")
                else:
                    print(f"[{task}::{split_name}] Test set size: {original_size} (max_test_samples={max_test_samples}, no truncation needed)")

            output_data = {
                "Definition": [TASK_SPECS[task]['definition']],
                "Positive Examples": [],
                "Negative Examples": [],
                "Instances": []
            }

            text_key = TASK_SPECS[task]['text_key']
            label_key = TASK_SPECS[task]['label_key']

            for example in dataset:
                input_text = _to_string(example.get(text_key))
                if task == 'CodeSearchNet':
                    output_text = CodeTaskPreprocessor._extract_first_paragraph(example.get(label_key))
                else:
                    output_text = _to_string(example.get(label_key))

                uid = hashlib.md5((task + "||" + input_text).encode("utf-8")).hexdigest()
                if use_instruction_pool:
                    instruction_input = _render_instruction(
                        task=task,
                        raw_input=input_text,
                        sample_key=f"{task}::{uid}",
                        split_name=split_name,
                        split_seed=split_seed,
                    )
                else:
                    definition = _to_string(TASK_SPECS[task]['definition']).strip()
                    instruction_input = f"{definition}\n{input_text}".strip()

                output_data["Instances"].append({
                    "id": f"{task}-{uid}",
                    "input": instruction_input,
                    "output": [output_text]
                })

            with open(os.path.join(save_dir, f"{split_name}.json"), "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️  Skipped {task}::{split_name}: {e}")


if __name__ == "__main__":
    np.random.seed(42)
    for split in ["train", "validation", "test"]:
        convert_to_codetask(
            split_name=split,
            split_seed=42,
            max_dev_samples=1000,
            max_test_samples=5000,
            max_train_samples=100000,
            use_instruction_pool=True,
        )
