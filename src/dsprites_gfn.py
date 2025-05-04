import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from gfn_attractors.data.dsprites import *
from gfn_attractors.misc import torch_utils as tu
from gfn_attractors.images import *
from gfn_attractors.dsprites import DSpritesAttractorsGFNEM, DSpritesGFNEM


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if args.dynamics:
        config = ImagesAttractorsGFNEMConfig.from_dict(config_dict)
    else:
        config = ImagesGFNEMConfig.from_dict(config_dict)
    config.seed = args.seed
    print(f"Successfully loaded config from {args.config}")
    print(config)

    save_dir = f'./rundata_dsprites/{args.run_name}'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    config.save_dir = save_dir
    config.save(f'{save_dir}/config.yaml')
    with open(save_dir + '/args.yaml', 'w') as f:
        f.write(yaml.dump(args))
    print(f"Saved config to {save_dir}/config.yaml")

    data_module = ContinuousDSpritesDataModule(batch_size=config_dict['data_batch_size'], 
                                               size=config_dict['data_size'],
                                               constant_orientation=True,
                                               min_scale=config_dict['data_min_scale'],
                                               holdout_xy_mode=config_dict['data_holdout_xy_mode'],
                                               holdout_xy_nonmode=config_dict['data_holdout_xy_nonmode'],
                                               holdout_xy_shape=config_dict['data_holdout_xy_shape'],
                                               holdout_xy_mode_color=config_dict['data_holdout_xy_mode_color'],
                                               holdout_shape_color=config_dict['data_holdout_shape_color'],
                                               seed=args.seed)
    data_module.prepare_data()
    
    if args.fixed_f and args.fixed_b:
        config.fixed_sd = -1
    
    print(f"Created data module with {len(data_module)} samples")

    if args.dynamics:
        model = DSpritesAttractorsGFNEM(config, data_module)
    else:
        model = DSpritesGFNEM(config, data_module)
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
        model.create_plots(gif_path=args.gif_path, n_items=args.n_items)
        return
    
    if not args.skip_vae:
        print("Training VAE")
        model.train_vae(config_dict['num_vae_updates'], config_dict['vae_batch_size'], 'vae.pt')

    print("Training GFN")
    name = f'{args.run_name}_' + ('dynamics' if args.dynamics else 'discretizer')

    if args.wandb:
        logger = WandbLogger(project='dsprites',
                                name=name,
                                entity='johank',
                                config={**config_dict, **vars(args)})
    else:
        logger = None

    trainer = pl.Trainer(max_epochs=args.epochs, 
                            devices=1 if device.type == "cpu" else [args.device],  
                            logger=logger,
                            num_sanity_val_steps=0,
                            enable_progress_bar=False)
    
    start_time = datetime.now()
    
    trainer.fit(model, tu.DummyDataset(1000), ckpt_path=args.ckpt)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    elapsed_time = datetime.now()-start_time
    print(f"Training time: {elapsed_time}")
    model.save(f'{save_dir}/{args.run_name}_{now}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=lambda x: x.lower() in ('true', '1'), default=False, help="If True, runs test")
    parser.add_argument("--config", type=str, help="Filepath to config file")
    parser.add_argument("--run_name", type=str, help="Name of run")
    parser.add_argument("--load", type=str, default=None, help="Filepath to load model from")
    parser.add_argument("--dynamics", type=lambda x: x.lower() in ('true', '1'), default=True, help="If True, trains attractor dynamics. Else, discretizer only.")
    parser.add_argument("--skip_vae", type=lambda x: x.lower() in ('true', '1'), default=False, help="If True, skips the VAE phase.")
    parser.add_argument("--device", type=int, help="Device index")
    parser.add_argument("--n_items", type=int, default=20, help="How many data entries to plot")
    parser.add_argument("--gif_path", type=str, default="pca_animation.gif", help="Filepath to save the created gif to")
    parser.add_argument("--save_dynamics_mlp", type=lambda x: x.lower() in ('true', '1'), default=False, help="Whether to save the forward- and backwards stepping MLPs")
    parser.add_argument("--fixed_f", type=str, default=None, help="Filepath to load custom fixed forward MLP from")
    parser.add_argument("--fixed_b", type=str, default=None, help="Filepath to load custom fixed backward MLP from")
    parser.add_argument("--ckpt", type=str, default=None, help="Filepath to load checkpoint from")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--wandb", type=lambda x: x.lower() in ('true', '1'), default=False, help="Whether to log to wandb")
    args = parser.parse_args()

    print(args)
    main(args)
