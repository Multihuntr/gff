from typing import Any
from gff.models.utae import ConvLayer
import torch.nn as nn

import numpy as np 
import pandas as pd

class LogisticRegression(nn.Module):
    def __init__(
        self,
        n_channels: int,
        out_channels: int,
        cond_dim: Any = None, # lead_time_embed_dim
    ):
        super().__init__(
        )

        # Logistic Regression implemented as a single convolutional layer with a kernel size of 3
        # self.model = ConvLayer(
        #     [n_channels, 1], k=3, p=1, padding_mode="reflect", last_relu=False, cond_dim=cond_dim
        # )
        self.model = nn.Conv2d(n_channels, out_channels, 3, padding=1, padding_mode='reflect')

    def forward(self, x, lead):
        return self.model(x)
        # return self.model.forward(x, lead)

def print_weights(model, feature_names):
    tensors = model.local_embed.model.weight.cpu().detach().numpy()
    nin, kh, kw, nout = tensors.shape
    weights = np.mean(tensors, axis=(1,2))
    df = pd.DataFrame(weights, columns=[f"class {i+1}" for i in range(nout)], index=feature_names)
    print(df)
