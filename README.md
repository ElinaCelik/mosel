# [NeurIPS 2026] Learning Pareto Stationary Fronts via Single-Pass Backpropagation

This is the official implementation for **MOSEL** (Multi-Objective Stackelberg Efficient Learning), a method for *a posteriori* multi-objective optimization in deep neural networks that **recovers a full front of Pareto stationary solutions at the computational cost of standard single-objective training**.

For more details, have a look at our paper:

```bibtex
@inproceedings{
    celik2026learning,
    title={Learning Pareto Stationary Fronts via Single-Pass Backpropagation},
    author={Elina Rojin Celik and Marcos Medeiros Raimundo and Isabel Valera},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026},
    url={https://openreview.net/forum?id=heKXlAkgm2}
}
```

## Repository overview

- ```configs/``` contains YAML configs with hyperparameters used for experiments in the paper 
- ```data/``` contains one folder per dataset; each readme.md explains how to download it
- ```multi_objective/```
    - ```main.py```: entry point, contains training, evaluation, checkpointing
    - ```eval_checkpoints.py```: re-evaluate saved checkpoints
    - ```settings.py```: default hyperparameters (generic, per method, per dataset)
    - ```objectives.py```: training objectives (cross-entropy, BCE, DEO fairness relaxation)
    - ```scores.py```: evaluation scores (loss, misclassification rate, DEO)
    - ```hv.py```: hypervolume computation
    - ```min_norm_solvers.py```: min-norm solver used by MGDA and ParetoMTL
    - ```utils.py```: dataset/model factories, seeding, Pareto front utilities
    - ```loaders/```: PyTorch datasets for all benchmarks
    - ```models/```: network architectures (LeNet, MLP, EfficientNet, ResNet, DistilBERT)
    - ```methods/```: implementation of MOSEL and all baselines, overview below




## Getting started

1. Create a venv:

```bash
python3 -m venv env
source env/bin/activate
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Download the datasets as described in `data/*/readme.md`.

## Usage

### Run the code

```bash
python multi_objective/main.py --dataset mm --method mosel
```

To reproduce the experiments from the paper, pass the respective config file from `configs/`. The configs are named `<dataset>_<method>.yaml` and contain the hyperparameter values used for the experiments in the paper. They override the defaults in `multi_objective/settings.py`. Dataset and method are still selected with `--dataset` and `--method`:

```bash
python multi_objective/main.py --dataset mm --method mosel --config configs/mm_mosel.yaml
```

Every setting can also be overridden from the command line with `--<setting> <value>`, which takes precedence over the config file. Lists are passed comma-separated. Run `python multi_objective/main.py --dataset mm --method mosel --help` to list all settings and their defaults for a given dataset and method.

```bash
python multi_objective/main.py --dataset mm --method mosel --config configs/mm_mosel.yaml --epochs 50 --lr 0.0001,0.001 --batch_size 128
```

For the logs and results see newly created folder `results`.

### Re-evaluate saved checkpoints 
Pass the same arguments as for training (dataset and method) as well as path to the checkpoint:

```bash
python multi_objective/eval_checkpoints.py --dataset mm --method mosel --checkpoint_dir results/mosel/multi_mnist/<run>/checkpoints
```

### Available datasets:

| command-line option  | Description                  |
|----------------------|------------------------------|
| `-d mm`              | Multi-MNIST dataset          |
| `-d mf`              | Multi-Fashion dataset        |
| `-d mfm`             | Multi-Fashion+MNIST dataset  |
| `-d adult`           | Adult dataset                |
| `-d compass`         | Compass dataset              |
| `-d credit`          | Default dataset              |
| `-d celeba`          | CelebA dataset               |
| `-d fairface`        | FairFace dataset             |
| `-d civil`           | CivilComments dataset        |

### Available algorithms:

| command-line option  | Description                              |
|----------------------|------------------------------------------|
| `-m mosel`           | MOSEL algorithm (ours)                   |
| `-m cosmos`          | COSMOS algorithm [1]                     | 
| `-m hyper_ln`        | PHN (Linear Scalarization) algorithm [2] | 
| `-m hyper_epo`       | PHN (EPO) algorithm [2]                  | 
| `-m pmtl`            | ParetoMTL algorithm [3]                  | 
| `-m mgda`            | MGDA algorithm [4]                       |
| `-m pcgrad`          | PCGrad algorithm [5]                     |
| `-m gradnorm`        | GradNorm algorithm [6]                   |
| `-m nashmtl`         | NashMTL algorithm [7]                    |
| `-m famo`            | FAMO algorithm [8]                       |
| `-m fair_classifier` | Fair classifier baseline [9]             |
| `-m SingleTask`      | Treat each objective as single task      | 
| `-m uniform`         | Uniform scaling of all objectives        | 

[1] Ruchte, M. and Grabocka, J., 2021. Scalable pareto front approximation for deep multi-
objective learning. In *2021 IEEE international conference on data mining (ICDM)*

[2] Navon, A., Shamsian, A., Chechik, G. and Fetaya, E., 2021. Learning the Pareto Front with Hypernetworks. In *International Conference on Learning Representations*

[3] Lin, X., Zhen, H.L., Li, Z., Zhang, Q. and Kwong, S., 2019. Pareto multi-task learning. *Advances in Neural Information Processing Systems*

[4] Sener, S. and Koltun, V., 2018. Multi-task learning as multi-objective optimization. *Advances in Neural Information Processing Systems*

[5] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K. and Finn, C., 2020. Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*

[6] Chen, Z., Badrinarayanan, V., Lee, C. and Rabinovich, A., 2018. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International conference on machine learning*

[7] Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G. and Fetaya, E., 2022. Multi-task learning as a bargaining game. In *International conference on machine learning*

[8] Liu, B., Feng, Y., Stone, P. and Liu, Q., 2023. Famo: Fast adaptive multitask optimization. *Advances in Neural Information Processing Systems*

[9] Hardt, M., Price, E. and Srebro, N., 2016. Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*


---
This repository is based on https://github.com/ruchtem/cosmos and has been extended and adapted for our purposes.

