import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import networkx as nx

import matplotlib.pyplot as plt

import numpy as np
import torch
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from gfn_attractors.data.branching_diffusion import *
from gfn_attractors.misc import torch_utils as tu
from gfn_attractors.models.gfn_em import GFNEMConfig
from gfn_attractors.models.attractors_gfn_em import AttractorsGFNEMConfig
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

    save_dir = f'./rundata/{args.run_name}'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    config.save_dir = save_dir
    config.save(f'{save_dir}/config.yaml')
    with open(save_dir + '/args.yaml', 'w') as f:
        f.write(yaml.dump(args))
    print(f"Saved config to {save_dir}/config.yaml")

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
    
    data_module.prepare_data()
    
    if args.fixed_f and args.fixed_b:
        config.fixed_sd = -1
    
    if args.dynamics:
        model = BinaryVectorAttractorsGFNEM(config, data_module)
    else:
        model = BinaryVectorGFNEM(config, data_module)
    
    if args.load:
        print(f"Loading model from {args.load}. Mismatching parameters:")
        print(tu.load_partial_state_dict(model, torch.load(args.load)))
        
    if args.fixed_f and args.fixed_b:
        print(f"Loading forward fixed MLP models from {args.fixed_f} and {args.fixed_b}.")
        model.dynamics_model.set_fixed_mlps(args.fixed_f, args.fixed_b)          

    model.init_optimizers()
    model.to(device)

    print("Performing sanity check")
    model.sanity_test(plot=False)
    
    if args.save_dynamics_mlp:
        torch.save(model.dynamics_model.forward_mlp.state_dict(), f'{save_dir}/{args.run_name}_forwardmlp.pt')
        torch.save(model.dynamics_model.backward_mlp.state_dict(), f'{save_dir}/{args.run_name}_backwardmlp.pt')

    if args.test:
        return
    
    print("Training VAE")
    model.train_vae(config_dict['num_vae_updates'], config_dict['vae_batch_size'], 'vae.pt')

    print("Training GFN")
    name = f'{args.run_name}_' + ('dynamics' if args.dynamics else 'discretizer')
    if args.wandb:
        logger = WandbLogger(project='diffusion',
                                name=name,
                                entity='johank',
                                #  tags=[experiment_name],
                                config={**config_dict, **vars(args)})
    else:
        logger = None

    trainer = pl.Trainer(max_epochs=args.epochs, 
                            devices=1 if device.type == "cpu" else [args.device], 
                            accelerator="cpu" if device.type == "cpu" else "gpu", 
                            logger=logger,
                            num_sanity_val_steps=0,
                            enable_progress_bar=False)
    
    
    start_time = datetime.now()
    
    trainer.fit(model, tu.DummyDataset(1000))

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    elapsed_time = datetime.now()-start_time
    print(f"Training time: {elapsed_time}")
    model.save(f'{save_dir}/{args.run_name}_{now}.pt')
    
    if args.save_m:
        torch.save(model.m_model.state_dict(), f'{save_dir}/{args.run_name}_{now}_forwardmlp.pt')
    
def visualize_graph(data_module):
    G = data_module.graph

    # Ensure each node has a 'subset' attribute corresponding to its generation
    node_levels = {node: G.nodes[node].get('gen', 0) for node in G.nodes}
    nx.set_node_attributes(G, node_levels, 'subset')

    # Use multipartite_layout with subset_key='subset'
    pos = nx.multipartite_layout(G, subset_key='subset')

    # Flip y-coordinates to place the root at the top
    for key in pos:
        pos[key][1] = -pos[key][1]

    # Draw the graph
    plt.figure(figsize=(10, 6))
    pos = hierarchy_pos(G)  # 0 = root node label

    nx.draw(G, pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=2000,
            font_size=10)
    plt.title("Hierarchical Graph with Root Node at the Top")
    plt.show()
    
def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.4, vert_loc=0, xcenter=0.5):
    """
    Return a dict {node: (x, y)} for a hierarchical layout.
    Works for any directed or undirected tree.
    """
    if root is None:
        root = next(iter(nx.topological_sort(G))) if G.is_directed() else list(G.nodes())[0]

    def _hierarchy_pos(node, left, right, depth, pos):
        mid = (left + right) / 2
        pos[node] = (mid, -depth * vert_gap + vert_loc)
        children = list(G.successors(node)) if G.is_directed() else list(G.neighbors(node))
        if children:
            step = (right - left) / len(children)
            for i, child in enumerate(children):
                _hierarchy_pos(child,
                                left + i * step,
                                left + (i + 1) * step,
                                depth + 1, pos)
        return pos

    return _hierarchy_pos(root, 0, width, 0, {})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=lambda x: x.lower() in ('true', '1'), default=False, help="If True, runs test")
    parser.add_argument("--config", type=str, help="Filepath to config file")
    parser.add_argument("--run_name", type=str, help="Name of run")
    parser.add_argument("--load", type=str, default=None, help="Filepath to load model from")
    parser.add_argument("--fixed_f", type=str, default=None, help="Filepath to load custom fixed forward MLP from")
    parser.add_argument("--fixed_b", type=str, default=None, help="Filepath to load custom fixed backward MLP from")
    parser.add_argument("--dynamics", type=lambda x: x.lower() in ('true', '1'), help="If True, trains attractor dynamics. Else, discretizer only.")
    parser.add_argument("--device", type=int, help="Device index")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--wandb", type=lambda x: x.lower() in ('true', '1'), default=False, help="Whether to log to wandb")
    parser.add_argument("--save_dynamics_mlp", type=lambda x: x.lower() in ('true', '1'), default=False, help="Whether to save the resulting forward MLP model separately")
    parser.add_argument("--save_m", type=lambda x: x.lower() in ('true', '1'), default=False, help="Whether to save the resulting m model separately")
    parser.add_argument("--n_children", type=int, default=2, help="How many children each data node has. Supported 2 and 3.")
    

    args = parser.parse_args()
    main(args)