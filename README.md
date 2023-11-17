# Generation of Hard 3-SAT Instances Using G2SAT

This repository contains the work developed during an internship at the [LAMSADE][lamsade] laboratory, Paris Dauphine University, in collaboration with [CentraleSupélec][cs]. The project focuses on adapting the G2SAT model[^g2sat], a Graph Neural Network (GNN) model, using reinforcement learning to generate hard instances of SAT problems. The goal is to produce instances that are challenging to solve, contributing to the field of complexity and computational hardness. We build on previous work[^sato] which used reinforcement learning to generate hard graph instances, but focus on the domain of SAT instances.

## Team

- Intern: Fernando Kurike Matsumoto
- Supervisors at LAMSADE: Benjamin Negrevergne, Florian Sikora, Florian Yger
- Supervisors at CentraleSupélec: Vincent Mousseau, Pascale Le Gall

## Directory Structure

- [notebooks](./notebooks): Jupyter notebooks for data analysis and visualization. Main notebooks:
  - [g2sat.ipynb](./notebooks/g2sat.ipynb): Provides an example of how to manually use the SAT graphs and the RL environment, serving as a practical guide for interacting with the model.
  - [report_plots.ipynb](./notebooks/report_plots.ipynb): Contains the plots and visualizations used in the final internship report and presentation, illustrating key findings and model performance.
  - [metrics.ipynb](./notebooks/metrics.ipynb): Dedicated to a separate analysis of the metrics, especially focusing on the hardness measures available for the Minisat solver.

- [src](./src): Source code including the GNN model implementation, training and evaluation scripts, and utility functions.
  - This directory houses the core Python scripts and modules for the G2SAT model, including the model architecture ([gnn_models](./src/gnn_models)), the environment and training procedure ([`generation`](./src/generation)), and the SAT solvers ([`solvers`](./src/solvers/)).

## Setup

### Pre-requisites

- Python 3.10 or higher
- Git

### Installation instructions

1. Clone the repository:

  ```sh
  git clone https://github.com/fernandokm/hard-instances
  ```

2. Navigate to the project directory:

  ```sh
  cd hard-instances
  ```

3. Install dependencies:

   - The main dependencies are listed in the [requirements.txt](requirements.txt). Additional dependencies required to run jupyter notebooks and linters/formatters are given in [requirements-dev.txt](requirements-dev.txt).

     ```sh
     pip install -r requirements.txt
     pip install -r requirements-dev.txt
     ```

   - Alternatively, use pipenv:

     ```sh
     pipenv install --dev
     ```

## Usage

This repository provides several scripts for generating, training, and evaluating models for creating hard instances of SAT problems (all in the [src](src) directory). **For a detailed description of each script's functionality and parameters, use the `--help` flag.**

The following are the main scripts provided:

- **[train_g2sat.py](src/train_g2sat.py)**: This script is responsible for training the generation model, allowing customization through various hyperparameters.

- **[generate_eval.py](src/generate_eval.py)**: This script evaluates the performance of the trained G2SAT model at different checkpoints, supporting a range of instance sizes and solvers.

- **[generate_templates.py](src/generate_templates.py)**: This script generates the training and test data.

- **[rerun_solver.py](src/rerun_solver.py)**: Designed to recompute training metrics for a model, this script is particularly useful for recalculating metrics under different CPU conditions and solver configurations.

### Example Workflow for Reproducing the Best Model

1. **Generate the training and test data**.

     ```sh
     # Create the output directory
     mkdir templates

     # Training templates (small)
     python src/generate_templates.py --num_vars=50 --num_clauses=210 --num_templates=32000 --multinomial --seed=0

     # Test templates (small and large)
     python src/generate_templates.py --num_vars=50 --num_clauses=210 --num_templates=32000 --multinomial --seed=100
     python src/generate_templates.py --num_vars=100 --num_clauses=420 --num_templates=32000 --multinomial --seed=100
     ```

     The templates will be saved to `50x210x3(multinomial)_32000_seed0.txt`, `50x210x3(multinomial)_32000_seed100.txt` and `100x420x3(multinomial)_32000_seed100.txt`.

2. **Train the model**. A gpu is automatically used if available. To use a specific device, pass the flag `--gpu_device='cuda:N'`.

     ```sh
     python src/train_g2sat.py \
         --num_vars=50 \
         --num_clauses=210 \
         --num_episodes=32000 \
         --metric=propagations \
         --template_file='templates/50x210x3(multinomial)_32000_seed0.txt.gz' \
         --checkpoint_freq=1000 \
         --eval_freq=1000 \
         --eval_repetitions=10 \
         --eval_file=templates/{50x210,100x420}x3'(multinomial)_100_seed100.txt.gz' \
         --allow_overlaps \
         --num_sampled_pairs=200 \
         --lr=2e-5 \
         --lr_decay=0.9999 \
         --seed=0
     ```

2. **Evaluate the model**. The flags `--num_cpus` and `--device` should be updated to match your machine.

   - For the task of instance generation:

     ```sh
     python src/generate_eval.py runs/SAGE/<logdir> \
         --num_vars={50,100,200}
         --alpha={3.00,3.50,3.75,4.00,4.10,4.20,4.30,4.40,4.50,4.60,4.70,4.80,4.90,5.00,5.25,5.50,6,7} \
         --solver={minisat22,cadical153} \
         --runs=150 \
         --num_cpus=26 \
         --device='cuda:0,cuda:1' \
         --multinomial_templates \
         --num_sampled_pairs=100000 \
         --checkpoint=32000 \
         --output="eval_multinomial.parquet"
     ```

   - For the task of instance augmentation:

     ```sh
     python src/generate_eval.py runs/SAGE/<logdir> \
         --num_vars={50,100} \
         --alpha=4.2 \
         --solver=minisat22 \
         --runs=1000 \
         --num_cpus=28 \
         --device='cuda:0,cuda:1' \
         --checkpoint=32000 \
         --num_sampled_pairs=100000 \
         --output=complexify.parquet \
         --complexify={0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10}
     ```

For further details about these scripts, run them with the `--help` flag.

### Instance Augmentation

In order to train the model for instance augmentation, steps 1 and 2 above should be changed to:

1. **Generate the training and test data**. Replace the `--multinomial` flag by `--complexify_min` and `--complexify_max`:

     ```sh
     # Create the output directory
     mkdir templates

     # Training templates (small)
     python src/generate_templates.py --num_vars=50 --num_clauses=210 --num_templates=32000 --complexify_min=0 --complexify_max=0.25 --seed=0
     
     # Test templates (small and large)
     python src/generate_templates.py --num_vars=50 --num_clauses=210 --num_templates=32000 --complexify_min=0 --complexify_max=0.25 --seed=100
     python src/generate_templates.py --num_vars=100 --num_clauses=420 --num_templates=32000--complexify_min=0 --complexify_max=0.25 --seed=100
     ```

2. **Train the model**. Change the file names:

     ```sh
     python src/train_g2sat.py \
         --num_vars=50 \
         --num_clauses=210 \
         --num_episodes=32000 \
         --metric=propagations \
         --template_file=templates/50x210x3(0.0,0.25)_32000_seed0.txt.gz \
         --checkpoint_freq=1000 \
         --eval_freq=1000 \
         --eval_repetitions=10 \
         --eval_file=templates/{50x210,100x420}x3'(0.0,0.25)_100_seed100.txt.gz' \
         --allow_overlaps \
         --num_sampled_pairs=200 \
         --lr=2e-5 \
         --lr_decay=0.9999 \
         --seed=0
     ```

3. **Evaluate the model**. No changes.

[lamsade]: https://www.lamsade.dauphine.fr/en.html
[cs]: https://www.centralesupelec.fr/en
[^g2sat]: Jiaxuan You et al. “G2SAT: Learning to Generate SAT Formulas”. In: Advances in
neural information processing systems 32 (2019), pp. 10552–10563. Github: <https://github.com/JiaxuanYou/G2SAT>.
[^sato]: Ryoma Sato, Makoto Yamada, and Hisashi Kashima. “Learning to Sample Hard Instances for Graph Algorithms”. In: Proceedings of The Eleventh Asian Conference on Machine Learning. Ed. by Wee Sun Lee and Taiji Suzuki. Vol. 101. Proceedings of Machine Learning Research. PMLR, 2019, pp. 503–518. Github: <https://github.com/joisino/HiSampler>.
