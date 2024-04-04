import pandas as pd
import numpy as np
import torch
from tqdm import trange
from tqdm import tqdm
import yaml
from datetime import date
import random

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Using config.yaml to get params
    with open("config.yaml", "r") as f:
        params = yaml.safe_load(f)

    params["device"] = 'cpu' if params["device"] == 123 else f"cuda:{params['device']}"
    params["exp_name"] = f"{params['exp_name']}_{params['dataset']}_{params['sentence_encoder']}_{date.today()}"

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    ## TODO: run the main func here ##

