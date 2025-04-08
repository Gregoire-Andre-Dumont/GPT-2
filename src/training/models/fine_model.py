"""Module that manages the pre-trained GPT-2 for fine-tuning."""

import torch
import torch.nn as nn
from transformers import GPT2Model
from peft import LoraConfig, get_peft_model, TaskType


class FineModel(nn.Module):
    """Pre-trained GPT-2 model for text classification."""

    def __init__(self, config):
        """Initialize the pre-trained torch layers."""

        super(FineModel, self).__init__()

        # Extract the parameters from the config
        self.model_name: int = config['model_name']
        self.num_classes: int = config['num_classes']
        self.max_length: int = config['max_length']
        self.hidden_size: int = config['hidden_size']
        self.lora_alpha: int = config['lora_alpha']
        self.lora_dropout: int = config['lora_dropout']
        self.lora_rank: int = config['lora_rank']

        # Initialize the pre-trained model and lora config
        self.back_bone = GPT2Model.from_pretrained(self.model_name)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout)

        # Apply LoRA to the model
        self.lora_model = get_peft_model(self.back_bone, lora_config)
        self.linear = nn.Linear(self.hidden_size * self.max_length, self.num_classes)

    def forward(self, input, targets=None):
        """Forward pass of the pre-trained GPT-2 model."""

        outputs = self.lora_model(input_ids=input, return_dict=True)
        last_hidden_state = outputs.last_hidden_state

        batch_size = last_hidden_state.shape[0]
        output = self.linear(last_hidden_state.view(batch_size, -1))

        # Check whether we have to compute the loss
        if targets is not None:
            loss = nn.CrossEntropyLoss(output, targets)
            return output, loss

        return output, None


