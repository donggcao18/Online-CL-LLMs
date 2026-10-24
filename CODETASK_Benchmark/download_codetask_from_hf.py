"""Restore prepared Hub configs as the JSON files consumed by src/cl_dataset.py."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from task_info import TASK_LIST


def restore_split(dataset, destination):
    if not len(dataset):
        raise ValueError(f'Cannot restore an empty split: {destination}')
    first = dataset[0]
    payload = {
        'Definition': first['definition'],
        'Positive Examples': json.loads(first['positive_examples']),
        'Negative Examples': json.loads(first['negative_examples']),
        'Instances': [
            {'id': row['id'], 'input': row['input'], 'output': row['outputs']}
            for row in dataset
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', default='main', help='Use a commit SHA for reproducible experiments')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'trimmed_30k'))
    parser.add_argument('--tasks', nargs='+', choices=TASK_LIST, default=TASK_LIST)
    args = parser.parse_args()

    for task in args.tasks:
        dataset = load_dataset(args.repo_id, name=task, revision=args.revision)
        for split in ('train', 'validation', 'test'):
            restore_split(dataset[split], Path(args.output_dir) / task / f'{split}.json')
            print(f'Restored {task}/{split}: {len(dataset[split])} examples')


if __name__ == '__main__':
    main()
