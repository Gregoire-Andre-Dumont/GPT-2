"""Module that manages the pre-trained GPT-2 models."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch.nn import functional as F


class OpenAI(nn.Module):
    """Manage the pre-trained GPT-2 model."""

    def __init__(self, **config: dict):
        """Initialize the torch layers."""

        super(OpenAI, self).__init__()

        # Load the pre-trained model
        model_name: str = config['model_name']
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, inputs, targets=None):
        """forward pass of the GPT-2 model."""

        outputs = self.model(inputs)
        logits = outputs.logits

        if targets is not None:
            new_targets = targets.view(-1)
            new_logits = logits.view(-1, logits.size(-1))

            loss = F.cross_entropy(new_logits, new_targets, ignore_index=-1)
            return logits, loss

        return logits, None

