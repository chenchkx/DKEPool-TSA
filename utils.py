import random
import numpy as np
import torch
import dgl
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def set_seed(args=None):
    seed = 1 if not args else args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def make_log_dir(model_name, dataset, subdir):
    # make and return
    model_name = model_name.lower()
    log_dir = os.path.join(f"./data/run_log/{dataset}", model_name, subdir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Data_Processing():
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self.data = data
        
    def fill_na(self):
        self.data = self.data.fillna(value=-1)

    def data_encoding(self):
        self.data = pd.get_dummies(self.data)
        # model = OneHotEncoder(categories=[3])
        # self.data = model.fit_transform(self.data)

    def run(self) -> pd.DataFrame:
        self.fill_na()
        self.data_encoding()
        return self.data


if __name__ == "__main__":
    df = pd.DataFrame([[1,2,3,'a'],[np.nan,0,5,'b']])
    df = Data_Processing(df).run()
    print(df)
    