import os
import json
import hashlib 
import numpy as np
from typing import Any
from datasets import load_dataset
from tqdm import tqdm

FOLDER_NAME = os.path.dirname(os.path.abspath(__file__))

TASK_LIST = [
    'CodeTrans',
    'CodeSearchNet',
    'BFP',
    'CONCODE',
    'TheVault_Csharp',
    'KodCode',
    'RunBugRun',
    'CoST',
]

TASK_SPECS = {
    # 'CONCODE': {
    #     'dataset_name': 'AhmedSSoliman/CodeXGLUE-CONCODE',
    #     'text_key': 'nl',
    #     'label_key': 'code',
    #     'definition': 'Generate Java code from the following English description: ',
    # },
    # 'CodeTrans': {
    #     'dataset_name': 'CM/codexglue_codetrans',
    #     'text_key': 'java',
    #     'label_key': 'cs',
    #     'definition': 'Translate the following Java code into C#: ',
    # },
    # 'BFP': {
    #     'dataset_name': 'ayeshgk/code_x_glue_cc_code_refinement_annotated',
    #     'text_key': 'buggy',
    #     'label_key': 'fixed',
    #     'definition': 'Refactor or improve the following Java code: ',
    # },
    # 'KodCode': {
    #     'dataset_name': 'KodCode/KodCode-V1-SFT-R1',
    #     'text_key': 'question',
    #     'label_key': 'solution',
    #     'definition': 'Generate Python code from the following description: ',
    # },
    # 'RunBugRun': {
    #     'dataset_name': 'ASSERT-KTH/RunBugRun-Final',
    #     'text_key': 'buggy_code',
    #     'label_key': 'fixed_code',
    #     'definition': 'Refactor or improve the following Ruby code: ',
    # },
    # 'CoST': {
    #     'dataset_name': 'semeru/code-text-python',
    #     'text_key': 'code',
    #     'label_key': 'docstring',
    #     'definition': 'Translate the following C++ code into C#: ',
    # },

    # 'CodeSearchNet': {
    #     'dataset_name': 'semeru/code-text-ruby',
    #     'text_key': 'code',
    #     'label_key': 'docstring',
    #     'definition': 'Summarize the following Ruby code into English: ',
    # },
    'TheVault_Csharp': {
        'dataset_name': 'Fsoft-AIC/the-vault-function',
        'text_key': 'code',
        'label_key': 'docstring',
        'definition': 'Summarize the following C# code into English: ',
    }
}

TRAIN_ONLY_TASKS = {
    'KodCode': {'val': 5000, 'test': 5000},
    'RunBugRun': {'val': 972, 'test': 1000},
}

HF_SPLIT_MAP = {
    'train': 'train',
    'dev': 'validation',
    'test': 'test',
}


def _split_train_only(dataset, task, split, split_seed=42):
    sizes = TRAIN_ONLY_TASKS[task]
    test_size = sizes['test']
    val_size = sizes['val']

    tmp = dataset.train_test_split(test_size=test_size, seed=split_seed)
    test_ds = tmp['test']

    tmp2 = tmp['train'].train_test_split(test_size=val_size, seed=split_seed)
    train_ds = tmp2['train']
    val_ds = tmp2['test']

    mapping = {'train': train_ds, 'dev': val_ds, 'test': test_ds}
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}' for train-only task '{task}'")
    return mapping[split]


def _load_task_split(task, split_name, split_seed=42):
    spec = TASK_SPECS[task]

    if task == 'TheVault_Csharp':
        split_map = {
            'train': ['train/small'],
            'dev': ['validation'],
            'test': ['test'],
        }
        dataset = load_dataset(
            spec['dataset_name'],
            languages=['c#'],
            split_set=split_map[split_name],
        )
        return dataset

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


def convert_to_codetask(split_name="train", split_seed=42):
    if split_name not in HF_SPLIT_MAP:
        raise ValueError(f"Unsupported split_name '{split_name}'. Use one of {list(HF_SPLIT_MAP.keys())}")

    for task in TASK_LIST:
        try:
            save_dir = os.path.join(FOLDER_NAME, task)
            os.makedirs(save_dir, exist_ok=True)
            dataset = _load_task_split(task, split_name, split_seed=split_seed)

            output_data = {
                "Definition": [TASK_SPECS[task]['definition']],
                "Positive Examples": [],
                "Negative Examples": [],
                "Instances": []
            }

            text_key = TASK_SPECS[task]['text_key']
            label_key = TASK_SPECS[task]['label_key']

            for example in tqdm(dataset, desc=f"Processing {task}::{split_name}"):
                input_text = _to_string(example.get(text_key))
                if task == 'CodeSearchNet':
                    output_text = CodeTaskPreprocessor._extract_first_paragraph(example.get(label_key))
                else:
                    output_text = _to_string(example.get(label_key))

                uid = hashlib.md5((task + "||" + input_text).encode("utf-8")).hexdigest()

                output_data["Instances"].append({
                    "id": f"{task}-{uid}",
                    "input": input_text,
                    "output": [output_text]
                })

            with open(os.path.join(save_dir, f"{split_name}.json"), "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️  Skipped {task}::{split_name}: {e}")


if __name__ == "__main__":
    np.random.seed(42)
    for split in ["train", "dev", "test"]:
        convert_to_codetask(split_name=split, split_seed=42)
