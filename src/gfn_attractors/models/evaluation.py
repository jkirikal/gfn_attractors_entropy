from abc import ABC, abstractmethod
import numpy as np
import torch
import pandas as pd
from plotnine import *


class EvaluationAttractorsModel(ABC):

    def __init__(self, model, data_module, seed=0):
        self.model = model
        self.data_module = data_module
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def config(self):
        return self.model.config
    
    @property
    def device(self):
        return self.model.device
    
    def sample_training_batch(self, n):
        """
        Should at least contain 'index' and 'x' tensors.
        """
        indices = self.rng.choice(self.data_module.train_indices, n, replace=True)
        batch = self.data_module.create_batch(indices)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch

    def sample_validation_batch(self, n):
        """
        Should at least contain 'index' and 'x' tensors.
        """
        indices = self.rng.choice(self.data_module.test_indices, n, replace=True)
        batch = self.data_module.create_batch(indices)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch

    
    def to(self, device):
        self.model = self.model.to(device) 

    def calculate_speed_and_distance(self, n=10000, single_point_n=100, single_point_batch_size=100, deterministic=False):
        batch_valid = self.sample_validation_batch(n)
        
        x = batch_valid['x']
        with torch.no_grad():
            z0 = self.model.get_z0(x)
            z_traj = self.model.sample_forward_trajectory(z0, deterministic=deterministic)
            w, _, _, _ = self.model.sample_w(z_traj, z0)
            z_hat = self.model.get_z_hat(w[:,-1])
            
        distances = (z_traj[:,1:] - z_traj[:,:-1]).norm(dim=-1)
        total_distance = distances.sum(dim=1)
        generated_point_distance = (z_traj[:,0] - z_hat).norm(dim=1)
        
        print("Relative 'speed': ",(total_distance/generated_point_distance).mean(0).item())
        print("Total trajectory distance: ", total_distance.mean(0).item())
        
        sampled_uniques = []
        for _ in range(single_point_n):
            single_point_batch = self.data_module.from_single_point(single_point_batch_size, prototype=True)

            single_x = single_point_batch['x'].to(self.device)

            with torch.no_grad():
                single_z0 = self.model.get_z0(single_x)
                single_z_traj = self.model.sample_forward_trajectory(single_z0, deterministic=False)
                single_w, _, _, _ = self.model.sample_w(single_z_traj, single_z0)
            
            sampled_uniques.append(len(set(self.model.m_model.stringify(single_w[:,-1]))))
        
        print(f"Mean number of unique sequences from same point per {single_point_n} samples: ", (sum(sampled_uniques) / len(sampled_uniques)))

    def plot_distances(self, n=30, deterministic=False):
        batch_train = self.sample_training_batch(n)
        batch_valid = self.sample_validation_batch(n)
        x = torch.cat([batch_train['x'], batch_valid['x']], dim=0)
        with torch.no_grad():
            z0 = self.model.get_z0(x)
            z_traj = self.model.sample_forward_trajectory(z0, deterministic=deterministic)
        speed = (z_traj[:,1:] - z_traj[:,:-1]).norm(dim=-1).view(n, 2, -1).mean(0).cpu().numpy()
        df = pd.DataFrame(speed.T, columns=['train', 'valid'])
        df['t'] = range(len(df))
        df = df.melt(id_vars='t', value_vars=['train', 'valid'], var_name='split', value_name='distance')

        p = (ggplot(df, aes(x='t', y='distance', color='split')) 
        + geom_vline(xintercept=self.config.num_steps, linetype='dashed', color='black')
        + geom_line(size=1)
        + labs(x='Step', y='Distance')
        + theme_bw()
        )
        return p, df
