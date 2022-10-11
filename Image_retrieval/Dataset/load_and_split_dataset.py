import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np


def split_dataset(path):
    data = pd.read_csv(path, header=None, names=[i for i in range(1801)])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, shuffle=True, random_state=2020)
    Xp_train = torch.from_numpy(np.array(X_train)).to(torch.float32)
    yp_train = torch.from_numpy(np.array(y_train)).to(torch.long)
    Xp_test = torch.from_numpy(np.array(X_test)).to(torch.float32)
    yp_test = torch.from_numpy(np.array(y_test)).to(torch.long)
    return Xp_train, yp_train, Xp_test, yp_test
