# GPT-2

This repository focuses on training different variants of the GPT-2 model from scratch, inspired by [Andrej Karpathy](https://github.com/karpathy).

## Getting started

This section contains the steps that need to be taken to get started with this project and fully reproduce our experiments.
This project was developed on Windows 11 os on Python 3.12 and Ubuntu on Python 3.11

### 1. Clone the repository

Make sure to clone the repository with your favourite git client or using the following command:

```
https://github.com/Gregoire-Andre-Dumont/GPT-2.git
```

### 2. Install the required packages

Install the required packages (on a virtual environment is recommended) using the following command:

```shell
pip install -r requirements.txt
```

### 3. Create Wandb account

This project uses Weights & Biases (wandb) to visualize and track training results. To use this functionality:

1. Create a free account at [wandb.ai](https://wandb.ai/site) if you don't already have one.
   
2. Modify the `setup/setup_wandb.py` file with your specific project name
   
3. Authenticate your wandb account by running in the terminal
   ```shell
   wandb login --relogin
   ```
   
4. Paste your API token when prompted. You can find your API token in your wandb account settings.

Once configured, the training runs will automatically log the metrics and visualizations to your wandb dashboard. This allows you to monitor the training progress, compare experiments, and share results with collaborators.

## Run train

`train.py` is the main script for training models. It reads a configuration file `conf/train.yaml` which specifies the model configuration and additional training parameters such as test size. The model selected in the `conf/train.yaml` can be found in the `conf/model` folder where the preprocessing, training and postprocessing steps are defined. When training is finished, the trained model is saved in the tm directory with a hash that depends on the model configurations. The best models are stored in the `best_model` folder.

## Run wandb sweeps

This project uses the sweep functionality from Wandb to perform hyperparameter optimization.

1. Create a sweep configuration in the wandb platform by. our swwep configuration is stored in `conf/sweep/gpt_nano`
   - you should define which parameters you want to optimize
   - you should specificy the ranges for each parameter
   - you should select the distribution type (uniform, log_uniform, categorical)
  
2. Once your sweep is configured in the wandb interface, you'll receive a command similar to
    ```shell
   wandb agent project_name/sweep_id
   ```
3. Paste and run the wandb agent command in your terminal to start the hyperparameter optimization process. You should make sure that your virtual environment is activated and uses python3 instead of python.





