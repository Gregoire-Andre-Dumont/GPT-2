"""Module that manages the pre-trained GPT-2 for fine-tuning."""

import torch
import torch.nn as nn
from transformers import GPT2Model

class FineModel(nn.Module):
    """Pre-trained GPT-2 model for text classification."""

    def __init__(self, config):
        """Initialize the pre-trained torch layers."""

        super(FineModel, self).__init__()

        # Extract the
        self.model_name: int = config['model_name']
        self.num_classes: int = config['num_classes']
        self.max_length: int = config['max_length']
        self.hidden_size: int = config['hidden_size']

        self.gpt2model = GPT2Model.from_pretrained(self.model_name)
        self.linear = nn.Linear(self.hidden_size * self.max_length, self.num_classes)

    def forward(self, input, targets=None):
        """Forward pass of the pre-trained GPT-2 model."""

        output, _ = self.gpt2model(input_ids=input)
        batch_size = output.shape[0]
        output = self.linear(output.view(batch_size, -1))

        # Check whether we have to compute the loss
        if targets is not None:
            loss = nn.CrossEntropyLoss(output, targets)
            return output, loss

        return output, None

