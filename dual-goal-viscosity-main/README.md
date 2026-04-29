# Learning Viscous Value Functions in Dual Goal Space

## Setup

```sh
conda create -y -n dgvisc python=3.10
conda activate dgvisc

pip install -U "jax[cuda12]"
pip install optax flax distrax
pip install Pillow tqdm rich ogbench wandb[meta] ml-collections matplotlib==3.7.5
```
