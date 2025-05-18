import os
import sys
import torch
import KE_model
import importlib
from utils import net_utils, path_utils
from configs.base_config import Config
import wandb
import torch.nn as nn
import random
import numpy as np
import pathlib
from copy import deepcopy
import csv
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.logging import AverageMeter, ProgressMeter
from utils.eval_utils import accuracy
from layers.CS_KD import KDLoss
from torch.utils.tensorboard import SummaryWriter
import logging


# Function to get training and validation functions from the specified trainer module
def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    return trainer.train, trainer.validate
    
# Function to train the model for a single generation
def train_dense(cfg, generation, model=None, fisher_mat=None):
    dataset = getattr(data, cfg.set)(cfg)
    if model is None:
        model = net_utils.get_model(cfg)
        if cfg.use_pretrain:
            net_utils.load_pretrained(cfg.init_path, cfg.gpu, model, cfg)

    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained, cfg.gpu, model, cfg)
        model = net_utils.move_model_to_gpu(cfg, model)
        if not cfg.no_reset:
            net_utils.split_reinitialize(cfg, model, reset_hypothesis=cfg.reset_hypothesis)
    
    model = net_utils.move_model_to_gpu(cfg, model)
    ###########################################################################
    # Generate importance data for AutoS in the first generation
    if generation == 0 and cfg.autos:
        print("Generate importance data for AutoS")
        pruner = Pruner(model, dataset.train_loader, cfg.device,cfg=cfg)
        importance_data_path = os.path.join(cfg.exp_dir, "importance_data.pkl")
        autos_model_path = os.path.join(cfg.exp_dir, "autos_model.pth")
        pruner.generate_importance_data(sparsity=cfg.sparsity, save_path=importance_data_path)
        print(f"cfg.exp_dir: {cfg.exp_dir}")
        print(f"autos_model_path: {cfg.autos_model_path}")
        train_autos_model(
            data_path=importance_data_path,
            save_path=autos_model_path,
            device=cfg.device,
            epochs=1,
            batch_size=64,
            lr=0.001,
            cfg=cfg
        )

    ###########################################################################   
    # Use get_directories for checkpoints and logs
    if cfg.save_model:
        run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg, generation)
        net_utils.save_checkpoint(
            {"epoch": 0, "arch": cfg.arch, "state_dict": model.state_dict()},
            is_best=False,
            filename=ckpt_base_dir / f"init_model.state",
            save=False
        )
    
    cfg.trainer = 'default_cls'
    cfg.pretrained = None
    
    if cfg.reset_important_weights:
        if cfg.autos or cfg.snip:
            ckpt_path, fisher_mat, model = KE_model.ke_cls_train_fish(cfg, model, generation, fisher_mat)
            pruner = Pruner(model, dataset.train_loader, cfg.device, silent=False, cfg=cfg)
            if cfg.autos:
                fisher_mat = pruner.autos_prune(1 - cfg.sparsity, autos_model_path=cfg.autos_model_path)
            else:
                fisher_mat = pruner.snip(1 - cfg.sparsity)
            sparse_model = net_utils.extract_sparse_weights(cfg, model, fisher_mat)
            tst_acc1, tst_acc5 = KE_model.ke_cls_eval_sparse(cfg, sparse_model, generation, ckpt_path, 'acc_pruned_model.csv')
            model = net_utils.reparameterize_non_sparse(cfg, model, fisher_mat)
            sparse_mask = fisher_mat
            torch.save(sparse_mask.state_dict(), os.path.join(cfg.exp_dir, f"mask_{'autos' if cfg.autos else 'snip'}_{generation}.pth"))
        else:
            ckpt_path, fisher_mat, model = KE_model.ke_cls_train_fish(cfg, model, generation, fisher_mat)
            sparse_mask = net_utils.extract_new_sparse_model(cfg, model, fisher_mat, generation)
            torch.save(sparse_mask.state_dict(), os.path.join(cfg.exp_dir, f"sparse_mask_{generation}.pth"))
            model = net_utils.reparameterize_non_sparse(cfg, model, sparse_mask)
        tst_acc1, tst_acc5 = KE_model.ke_cls_eval_sparse(cfg, model, generation, ckpt_path, 'acc_drop_reinit.csv')
    
        if cfg.freeze_fisher:
            model = net_utils.diff_lr_sparse(cfg, model, sparse_mask)
            print('freezing the important parameters')
    else:
        ckpt_base_dir, model = KE_model.ke_cls_train(cfg, model, generation)
        sparse_mask = net_utils.create_dense_mask_0(deepcopy(model), cfg.device, value=1)  # 
    
    non_overlapping_sparsemask = net_utils.create_dense_mask_0(deepcopy(model), cfg.device, value=0)
    
    return model, fisher_mat, sparse_mask

# Function to calculate the percentage overlap between previous and current masks
def percentage_overlap(prev_mask, curr_mask, percent_flag=False):
    total_percent = {}
    for (name, prev_parm_m), curr_parm_m in zip(prev_mask.named_parameters(), curr_mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            overlap_param = ((prev_parm_m == curr_parm_m) * curr_parm_m).sum()
            assert torch.numel(prev_parm_m) == torch.numel(curr_parm_m)
            N = torch.numel(prev_parm_m.data)
            if percent_flag:
                no_of_params = ((curr_parm_m == 1) * 1).sum()
                percent = overlap_param / no_of_params
            else:
                percent = overlap_param / N
            total_percent[name] = (percent * 100)
    return total_percent

# Main function to start the Knowledge Evolution process
def start_KE(cfg):
    cfg.exp_dir = os.path.join(os.getcwd(), 'experiments', 'autos_cifar10')
    os.makedirs(cfg.exp_dir, exist_ok=True)
    base_dir = pathlib.Path(f"{path_utils.get_checkpoint_dir()}/{cfg.name}")
    os.makedirs(base_dir, exist_ok=True)

    ckpt_queue = []
    model = None
    fish_mat = None

    #############ResNet#################
    weights_history = {
        'conv1': [], 'layer1.0.conv1': [], 'layer2.0.conv1': [],
        'layer3.0.conv1': [], 'layer4.0.conv1': [], 'fc': []
    }
    mask_history = {}
    all_epoch_data = []  # Store epoch_metrics for all generations

    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0
        model, fish_mat, sparse_mask = train_dense(cfg, gen, model=model, fisher_mat=fish_mat)
        weights_history['conv1'].append(model.conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer1.0.conv1'].append(model.layer1[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer2.0.conv1'].append(model.layer2[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer3.0.conv1'].append(model.layer3[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer4.0.conv1'].append(model.layer4[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['fc'].append(model.fc.weight.data.clone().cpu().numpy().flatten())
        
        mask_history[gen] = {}
        if sparse_mask is not None:
            for name, param in sparse_mask.named_parameters():
                if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                    mask_history[gen][name] = param.data.clone().cpu().numpy()
            print(f"Generation {gen}: Stored mask layers: {list(mask_history[gen].keys())}")
        else:
            print(f"Generation {gen}: No sparse mask generated")
        
        if cfg.num_generations == 1:
            break

    # Plotting with enhanced visualizations
    if mask_history and len(mask_history) > 0:
        # Prepare sparsity data
        sparsity_data = []
        for gen in mask_history:
            for layer_name, mask in mask_history[gen].items():
                sparsity = 100 * (1 - mask.mean())
                sparsity_data.append({'Generation': gen, 'Layer': layer_name, 'Sparsity': sparsity})
        sparsity_df = pd.DataFrame(sparsity_data)

        # 1. Sparsity Bar Plot (Seaborn)
        plt.figure(figsize=(14, 6))
        sns.barplot(x='Layer', y='Sparsity', hue='Generation', data=sparsity_df, palette='viridis')
        plt.title(f'Sparsity Across Layers for {cfg.set}, {cfg.arch}', fontsize=14, pad=15)
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Sparsity (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.legend(title='Generation', fontsize=10, title_fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'sparsity_across_layers.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Sparsity Line Plot (Plotly)
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Sparsity Over Generations ({cfg.set}, {cfg.arch})'])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for layer_name in sparsity_df['Layer'].unique():
            layer_df = sparsity_df[sparsity_df['Layer'] == layer_name]
            fig.add_trace(go.Scatter(
                x=layer_df['Generation'],
                y=layer_df['Sparsity'],
                mode='lines+markers',
                name=layer_name,
                line=dict(width=2, color=colors[len(fig.data) % len(colors)]),
                marker=dict(size=8)
            ))
        fig.update_layout(
            xaxis_title='Generation',
            yaxis_title='Sparsity (%)',
            showlegend=True,
            template='plotly_white',
            font=dict(size=12),
            legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)', bordercolor='Black', borderwidth=1),
            width=800,
            height=500
        )
        fig.write_html(os.path.join(base_dir, 'sparsity_over_generations.html'))
        fig.write_image(os.path.join(base_dir, 'sparsity_over_generations.png'), width=800, height=500)

        # 3. Mask Overlap Plot (Seaborn)
        overlap_data = []
        for gen1 in mask_history:
            for gen2 in mask_history:
                if gen1 < gen2:
                    prev_model = deepcopy(model)
                    curr_model = deepcopy(model)
                    for (name, param) in prev_model.named_parameters():
                        if name in mask_history[gen1]:
                            param.data = torch.from_numpy(mask_history[gen1][name]).to(param.device)
                    for (name, param) in curr_model.named_parameters():
                        if name in mask_history[gen2]:
                            param.data = torch.from_numpy(mask_history[gen2][name]).to(param.device)
                    overlap = percentage_overlap(prev_model, curr_model, percent_flag=True)
                    for layer, perc in overlap.items():
                        overlap_data.append({'Layer': layer, 'Comparison': f'Gen {gen1} vs Gen {gen2}', 'Overlap': perc})
        overlap_df = pd.DataFrame(overlap_data)
        if not overlap_df.empty:
            plt.figure(figsize=(14, 6))
            sns.barplot(x='Layer', y='Overlap', hue='Comparison', data=overlap_df, palette='magma')
            plt.title(f'Mask Overlap Between Generations ({cfg.set}, {cfg.arch})', fontsize=14, pad=15)
            plt.xlabel('Layer', fontsize=12)
            plt.ylabel('Overlap (%)', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.legend(title='Comparison', fontsize=10, title_fontsize=12)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, 'mask_overlap.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Weights Distribution Plot (Seaborn)
        for layer_name, weights_list in weights_history.items():
            plt.figure(figsize=(12, 6))
            for gen, weights in enumerate(weights_list):
                sns.kdeplot(weights[:100], label=f'Generation {gen}', linewidth=2, alpha=0.7)
            plt.title(f'Weight Distribution for {layer_name} ({cfg.set}, {cfg.arch})', fontsize=14, pad=15)
            plt.xlabel('Weight Value', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(fontsize=10, title='Generation', title_fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, f"{layer_name}_weights_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # Log if no mask history
        if not mask_history:
            print("No mask history available to plot")
# Function to clean up checkpoint directory
def clean_dir(ckpt_dir, num_epochs):
    if '0000' in str(ckpt_dir):
        return
    rm_path = ckpt_dir / 'model_best.pth'
    if rm_path.exists():
        os.remove(rm_path)
    rm_path = ckpt_dir / f'epoch_{num_epochs - 1}.state'
    if rm_path.exists():
        os.remove(rm_path)
    rm_path = ckpt_dir / 'initial.state'
    if rm_path.exists():
        os.remove(rm_path)

# Main execution block
if __name__ == '__main__':
    cfg = Config().parse(None)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.conv_type = 'SplitConv'
    
    if not cfg.no_wandb:
        if len(cfg.group_vars) > 0:
            if len(cfg.group_vars) == 1:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
            else:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
                for var in cfg.group_vars[1:]:
                    group_name = group_name + '_' + var + str(getattr(cfg, var))
            wandb.init(project="llf_ke", group=cfg.group_name, name=group_name)
            for var in cfg.group_vars:
                wandb.config.update({var: getattr(cfg, var)})
                
    if cfg.seed is not None and cfg.fix_seed:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    start_KE(cfg)

###$ python .\DNR\train_KE_cls.py  --weight_decay 0.0001 --arch Split_ResNet18 --no_wandb --set CIFAR10 --data /data/input-ai/datasets/cifar10 \
          ## --epochs 200 --num_generations 11  --sparsity 0.8 --save_model --snip --reset_important_weights
