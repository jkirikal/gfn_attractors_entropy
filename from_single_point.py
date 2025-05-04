import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

import sys
import os
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
sys.path.append("C:\\Users\\johankir\\OneDrive - Tartu Ülikool\\Dokumendid\\semester_6\\lõputöö\\gfn_attractors\\src")

import numpy as np
import torch
from dataclasses import asdict
import yaml
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from gfn_attractors.data.branching_diffusion import *
from gfn_attractors.misc import torch_utils as tu
from gfn_attractors.models.gfn_em import GFNEM, GFNEMConfig
from gfn_attractors.models.evaluation import EvaluationAttractorsModel
from gfn_attractors.models.attractors_gfn_em import AttractorsGFNEM, AttractorsGFNEMConfig
from gfn_attractors.binary_vectors import *

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if args.dynamics:
        config = AttractorsGFNEMConfig.from_dict(config_dict)
    else:
        config = GFNEMConfig.from_dict(config_dict)
    print(f"Successfully loaded config from {args.config}")
    print(config)

    if args.n_children == 2:
        data_module = BinarySplitDataModule(depth=config_dict['data_depth'], 
                                            repeat=config_dict['data_repeat'], 
                                            sample_ancestors=config_dict['data_sample_ancestors'],
                                            min_test_depth=config_dict['data_min_test_depth'],
                                            batch_size=config_dict['data_batch_size'],
                                            seed=args.seed)
    elif args.n_children == 3:
        data_module = TernarySplitDataModule(depth=config_dict['data_depth'], 
                                            repeat=config_dict['data_repeat'], 
                                            sample_ancestors=config_dict['data_sample_ancestors'],
                                            min_test_depth=config_dict['data_min_test_depth'],
                                            batch_size=config_dict['data_batch_size'],
                                            seed=args.seed)
    else:
        print("Data module n_children amount unsupported")
        return
    
    if args.fixed_f and args.fixed_b:
        config.fixed_sd = -1
    
    if args.dynamics:
        model = BinaryVectorAttractorsGFNEM(config, data_module)
    else:
        model = BinaryVectorGFNEM(config, data_module)
        
    if args.load:
        print(f"Loading model from {args.load}. Mismatching parameters:")
        print(tu.load_partial_state_dict(model, torch.load(args.load)))
    else:
        print("You should choose a model")
        return
    
    if args.fixed_f and args.fixed_b:
        print(f"Loading forward fixed MLP models from {args.fixed_f} and {args.fixed_b}.")
        model.dynamics_model.set_fixed_mlps(args.fixed_f, args.fixed_b)
    
    
    
    model.to(device)

    
    evaluation = EvaluationAttractorsModel(model, data_module, args.seed)
    evaluation.to(device)
    
    model.create_plots(gif_path=f'animations/samepoint_{args.content_name}.gif', n_items=args.n_items, same_point=True)
        
    #model.sanity_test()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Filepath to config file")
    parser.add_argument("--load", type=str, default=None, help="Filepath to load model from")
    parser.add_argument("--fixed_f", type=str, default=None, help="Filepath to load custom fixed forward MLP from")
    parser.add_argument("--fixed_b", type=str, default=None, help="Filepath to load custom fixed backward MLP from")
    parser.add_argument("--dynamics", type=lambda x: x.lower() in ('true', '1'), help="If True, trains attractor dynamics. Else, discretizer only.")
    parser.add_argument("--device", type=int, help="Device index")
    parser.add_argument("--content_name", type=str, default="gfn_evaluation", help="Filepath to save the created gif to")
    parser.add_argument("--n_children", type=int, default=2, help="How many children each data node has. Supported 2 and 3.")
    parser.add_argument("--n_items", type=int, default=20, help="How many data entries to plot")
    
    parser.add_argument("--plot_gif", type=lambda x: x.lower() in ('true', '1'), default=False, help="Whether to create gif")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--wandb", type=lambda x: x.lower() in ('true', '1'), default=False, help="Whether to log to wandb")

    args = parser.parse_args()
    main(args)