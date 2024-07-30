import torch.nn as nn

import numpy as np
import pandas as pd


class LogisticRegression(nn.Module):
    """
    Logistic Regression implemented as a single convolutional layer with a kernel size of 3
    """

    def __init__(self, n_channels: int, out_channels: int):
        super().__init__()

        self.model = nn.Conv2d(n_channels, out_channels, 3, padding=1, padding_mode="reflect")

    def forward(self, x, lead):
        return self.model(x.squeeze(1))


def print_weights(model, feature_names):
    tensors = model.local_embed.model.weight.cpu().detach().numpy()
    nin, kh, kw, nout = tensors.shape
    weights = np.mean(tensors, axis=(1, 2))
    df = pd.DataFrame(weights, columns=[f"class {i+1}" for i in range(nout)], index=feature_names)
    print(df)
