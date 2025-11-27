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
