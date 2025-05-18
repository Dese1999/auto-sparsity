import os
import torch
import KE_model
import importlib
from utils import net_utils
from utils import path_utils
from configs.base_config import Config
import wandb
import random
import numpy as np
import pathlib
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
from utils.pruning import Pruner
from utils.net_utils import train_autos_model
import data

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
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    ckpt_queue = []
    model = None
    fish_mat = None
    

   #############ResNet#################
    weights_history = {
        'conv1': [],                  # Initial conv layer
        'layer1.0.conv1': [],         # First conv in the first block of layer1
        'layer2.0.conv1': [],         # First conv in the first block of layer2
        'layer3.0.conv1': [],         # First conv in the first block of layer3
        'layer4.0.conv1': [],         # First conv in the first block of layer4
        'fc': []                      # Final fully connected layer
    }
    mask_history = {}
    # Optional: Print model structure to verify layer names
  
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

    if mask_history and len(mask_history) > 0:
        plt.figure(figsize=(15, 10))
        any_data_plotted = False
        available_layers = mask_history[0].keys() if 0 in mask_history else []
        print(f"Available layers in mask_history: {list(available_layers)}")
        
        for layer_name in available_layers:
            sparsity_per_gen = []
            for gen in range(cfg.num_generations):
                if gen in mask_history and layer_name in mask_history[gen]:
                    mask = mask_history[gen][layer_name]
                    sparsity = 100 * (1 - mask.mean())
                    sparsity_per_gen.append(sparsity)
                else:
                    sparsity_per_gen.append(0)
            
            if any(sparsity_per_gen):
                plt.plot(range(cfg.num_generations), sparsity_per_gen, label=f'{layer_name}', marker='o')
                any_data_plotted = True
        
        if any_data_plotted:
            plt.title("Sparsity Changes Across Generations for Different Layers")
            plt.xlabel("Generation")
            plt.ylabel("Sparsity (%)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, "mask_sparsity_plot.png"))
            plt.show()
        else:
            print("No data to plot for mask sparsity")
    else:
        print("No mask history available to plot")

    for layer_name, weights_list in weights_history.items():
        plt.figure(figsize=(12, 5))
        for gen, weights in enumerate(weights_list):
            plt.plot(weights[:10], label=f'Generation {gen}', alpha=0.7)
        plt.title(f"Changes in {layer_name} Weights Across Generations")
        plt.xlabel("Weight Index")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_dir, f"{layer_name}_weights_plot.png"))
        plt.show()

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
