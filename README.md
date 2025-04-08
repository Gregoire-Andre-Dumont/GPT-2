# GPT-2

This repository focuses on training different variants of the GPT-2 model from scratch, inspired by [Andrej Karpathy](https://github.com/karpathy).

## Getting started

This section contains the steps that need to be taken to get started with this project and fully reproduce our experiments.
This project was developed on Windows 11 os on Python 3.12 and Ubuntu on Python 3.11

### 1. Clone the repository

Make sure to clone the repository with your favourite git client or using the following command:

```
git clone https://github.com/Gregoire-Andre-Dumont/GPT-2.git
```

### 2. Install the required packages

Install the required packages (on a virtual environment is recommended) using the following command:

```shell
pip install -r requirements.txt
```

### 3. Create Wandb account
This project uses Weights & Biases (Wandb) to visualize and track training results. To use this feature, you should create a free account at [wandb.ai](https://wandb.ai/site) if you don't already have one.  Next, modify the setup/setup_wandb.py file with your specific project name. Then you need authenticate your wandb account by running the following command in the terminal and paste your API token. You can find this token in your wandb account settings. Once configured, the training runs will automatically log the metrics and visualizations to your wandb dashboard. This allows you to monitor the training progress, compare experiments, and share results with collaborators.

```shell
wandb login --relogin
```


## Run train

`train.py` is the main script for training models. It reads a configuration file `conf/train.yaml` which specifies the model configuration and additional training parameters such as test size. The model selected in the `conf/train.yaml` can be found in the `conf/model` folder where the preprocessing, training and postprocessing steps are defined. When training is finished, the trained model is saved in the tm directory with a hash that depends on the model configurations. The best models are stored in the `best_model` folder.

## Run wandb sweeps

This project uses the sweep functionality from Wandb to perform hyperparameter optimization. On the Wandb platform, you should create a sweep configuration where you specify your parameters, their ranges and distributions. Our default sweep configuration is stored in the conf/sweep/gpt_nano.yaml file. To start a sweep, you should copy your wandb agent command from the sweep overview and run it in your terminal. Before doing so, you should make sure that your virtual environment is activated and use python3 instead of python when executing scripts to avoid compatibility issues. 

 ```shell
wandb agent project_name/sweep_id
```





