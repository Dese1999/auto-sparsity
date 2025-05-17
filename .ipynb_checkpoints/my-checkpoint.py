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
import os
import matplotlib.pyplot as plt
from models.split_resnet import binarize
from data.datasets import load_dataset
#content/DNR/DNR/data/datasets.py
# Function to get training and validation functions from the specified trainer module
def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    return trainer.train, trainer.validate

# Function to train the model for a single generation
def train_dense(cfg, generation, model=None, fisher_mat=None, train_loader=None, val_loader=None):
    """
    Train the model for one generation with learnable masks.
    Args:
        cfg: Configuration object
        generation: Current generation number
        model: Model to train (or None to create new)
        fisher_mat: Fisher matrix (optional)
        train_loader: Training data loader
        val_loader: Validation data loader
    Returns:
        model: Trained model
        fisher_mat: Updated Fisher matrix
        sparse_mask: Current mask
    """
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

    # Initialize masks with SNIP if at the start of a SNIP interval
    if generation % cfg.snip_interval == 0:
        model = net_utils.initialize_learnable_masks_with_snip(cfg, model, train_loader,fisher_mat)
        cfg.logger.info(f"Generation {generation}: Initialized masks with SNIP")

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
        ckpt_path, fisher_mat, model = KE_model.ke_cls_train_fish(cfg, model, generation, fisher_mat)
        sparse_mask = net_utils.extract_new_sparse_model(cfg, model, fisher_mat, generation)
        model = net_utils.reparameterize_non_sparse(cfg, model, model.binary_masks)
        torch.save(model.binary_masks.state_dict(), os.path.join(cfg.exp_dir, f"mask_{generation}.pth"))
        #####################################################################################################
        fisher_mat_np = {name: tensor.cpu().detach().numpy() for name, tensor in fisher_mat.items()}
        np.save(os.path.join(cfg.exp_dir, f"FIM_{generation}.npy"), fisher_mat_np)
        #np.save(os.path.join(cfg.exp_dir, f"FIM_{generation}.npy"), fisher_mat.cpu().detach().numpy())
        tst_acc1, tst_acc5 = KE_model.ke_cls_eval_sparse(cfg, model, generation, ckpt_path, 'acc_drop_reinit.csv')
        
        if cfg.freeze_fisher:
            model = net_utils.diff_lr_sparse(cfg, model, sparse_mask)
            print('freezing the important parameters')
    else:
        ckpt_base_dir, model = KE_model.ke_cls_train(cfg, model, generation)
        sparse_mask = net_utils.create_dense_mask_0(deepcopy(model), cfg.device, value=1)
    
    return model, fisher_mat, model.binary_masks


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

def start_KE(cfg):
    """
    Start Knowledge Evolution training with learnable masks and periodic SNIP reinitialization.
    """
    base_dir = pathlib.Path(f"{path_utils.get_checkpoint_dir()}/{cfg.name}")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    train_loader, val_loader = load_dataset(
        name=cfg.set,  
        root=cfg.data, 
        sample='default',  
        batch_size=cfg.batch_size
    )
    ckpt_queue = []
    model = None
    fish_mat = None

    # Assume train_loader and val_loader are defined
    # train_loader = ...  # Replace with actual loader
    # val_loader = ...    # Replace with actual loader

    weights_history = {
        'conv1': [],
        'layer1.0.conv1': [],
        'layer2.0.conv1': [],
        'layer3.0.conv1': [],
        'layer4.0.conv1': [],
        'fc': []
    }
    mask_history = {}

    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0
        model, fish_mat, sparse_mask = train_dense(
            cfg, gen, model=model, fisher_mat=fish_mat, train_loader=train_loader, val_loader=val_loader
        )
        weights_history['conv1'].append(model.conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer1.0.conv1'].append(model.layer1[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer2.0.conv1'].append(model.layer2[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer3.0.conv1'].append(model.layer3[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['layer4.0.conv1'].append(model.layer4[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history['fc'].append(model.fc.weight.data.clone().cpu().numpy().flatten())

        mask_history[gen] = {}
        if sparse_mask is not None:
            for idx, mask in enumerate(sparse_mask):
                mask_history[gen][f'mask_{idx}'] = mask.data.clone().cpu().numpy()
            print(f"Generation {gen}: Stored mask layers: {list(mask_history[gen].keys())}")
        else:
            print(f"Generation {gen}: No sparse mask generated")

        if cfg.num_generations == 1:
            break

    # Plotting logic remains unchanged
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
