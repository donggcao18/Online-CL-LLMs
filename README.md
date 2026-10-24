# Online-CL-LLMs

> This repository contains the official implementation for our ICML 2025 paper: *Exploiting Presentative Feature Distributions for Parameter-Efficient Continual Learning of Large Language Models*.

![Architecture](./Architecture.png)

## Requirements
* Python 3.10.12
* PyTorch 2.1.0
* Transformers 4.30.2
* CUDA 12.2

## Preparation
1. Setting up env
```sh
conda create -y -n nlp python=3.10.12
conda activate nlp 
cd Online-CL-LLMs
pip install -r requirements_v2.txt
```

2. Generating data for CodeTask dataset 
```sh
python CODETASK_Benchmark/parse_into_json.py
```

This prepares all eight tasks in `CODETASK_Benchmark/trimmed_30k/`.
Training sets above 30,000 examples are randomly sampled with seed 42;
smaller sets are kept in full. Selected IDs and rendered instructions are saved
so every task-order experiment can reuse the same data. Each source task is
loaded once for all its splits (TheVault loads each source split once).
The existing evaluation policy is preserved: at most 1,000 validation and 5,000
test examples, with the same split seed and selection as before. To retain full
evaluation splits instead, pass `--max-dev-samples 0 --max-test-samples 0`.
Use `--output-dir`, `--max-train-samples`, `--seed`, or `--tasks` to customize a run.
Preprocessing stops on errors rather than silently producing a partial benchmark.

Upload the prepared data once (after `huggingface-cli login`):

```sh
python CODETASK_Benchmark/upload_codetask_to_hf.py --repo-id YOUR_USERNAME/codetask-30k --private
```

Each task is a separate Hub configuration with train/validation/test splits.
The upload reads only the prepared JSON files; it does not reload source datasets
or rerun preprocessing. To upload a different prepared directory, use `--data_root`.

On a training machine, download and restore these prepared files once:

```sh
python CODETASK_Benchmark/download_codetask_from_hf.py --repo-id YOUR_USERNAME/codetask-30k
DATA_DIR=CODETASK_Benchmark/trimmed_30k bash qwen_coder_permutations.sh 1
```

For SLURM, use
`sbatch --export=ALL,DATA_DIR=CODETASK_Benchmark/trimmed_30k qwen_coder_permutations.sh`.
Other training scripts should set `--data_dir CODETASK_Benchmark/trimmed_30k`.
No source preprocessing is required on subsequent training runs. Private repos
require a logged-in account with access. Pin `--revision COMMIT_SHA` when restoring
to reuse a specific dataset version. You can also load a processed task directly:

```python
from datasets import load_dataset
task = load_dataset("YOUR_USERNAME/codetask-30k", "CONCODE")
```

3. Config for CodeTask has been already created and stored in  `configs/CodeTask`


4. And the generated pseudo data points are in `/generated_data`.

## Training

To implement T5 model on the CodeTask benchmark:

```sh
bash t5_normal.sh
```


<!--
## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{,
  title={Exploiting Presentative Feature Distributions for Parameter-Efficient Continual Learning of Large Language Models},
  author={Xin Cheng, Jiabo Ye, Haiyang Xu, Ming Yan, Ji Zhang, Feng Liu, Fei Huang, Lei Feng},
  booktitle={Proceedings of the 42th International Conference on Machine Learning (ICML'25)},
  year={2025}
}
```
-->

## Credits
The code of this repository partly relies on [SAPT](https://github.com/circle-hit/SAPT.git) and we would like to show our sincere gratitude to authors of it.
