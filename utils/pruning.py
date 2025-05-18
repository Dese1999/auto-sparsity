import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch
from copy import deepcopy
import utils
from utils.net_utils import create_dense_mask_0
from utils.autosnet import  ResNet18
from utils.autosnet import MLP 

class Pruner:
    def __init__(self, model, loader=None, device='cpu', silent=False):
        self.device = device
        self.loader = loader
        self.model = model
        
        self.weights = [layer for name, layer in model.named_parameters() if 'mask' not in name]
        self.indicators = [torch.ones_like(layer) for name, layer in model.named_parameters() if 'mask' not in name]
        self.mask_ = utils.net_utils.create_dense_mask_0(deepcopy(model), self.device, value=1)
        self.pruned = [0 for _ in range(len(self.indicators))]
 
        if not silent:
            print("number of weights to prune:", [x.numel() for x in self.indicators])

    def indicate(self):
        for weight, indicator in zip(self.weights, self.indicators):
            weight.data = weight * indicator
    
    def snip(self, sparsity, mini_batches=1, silent=False):
        mini_batches = len(self.loader) / 32
        mini_batch = 0
        self.indicate()
        self.model.zero_grad()
        grads = [torch.zeros_like(w) for w in self.weights]
        
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            x = self.model.forward(x)
            L = torch.nn.CrossEntropyLoss()(x, y)
            grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g)) 
                     for g, ag in zip(grads, torch.autograd.grad(L, self.weights, allow_unused=True))]
            
            mini_batch += 1
            if mini_batch >= mini_batches: 
                break

        with torch.no_grad():
            saliences = [(grad * weight).view(-1).abs().cpu() for weight, grad in zip(self.weights, grads)]
            saliences = torch.cat(saliences)
            
            thresh = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0])
            
            for j, layer in enumerate(self.indicators):
                layer[(grads[j] * self.weights[j]).abs() <= thresh] = 0
                self.pruned[j] = int(torch.sum(layer == 0))
        
        idx = 0
        for name, param in self.mask_.named_parameters():
            if 'mask' not in name:
                param.data = self.indicators[idx]
                idx = idx + 1
                continue

        self.model.zero_grad() 
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])

        return self.mask_

    def generate_importance_data(self, sparsity, save_path="data.pkl", num_iterations=20, num_batches=10):
        """
        Generate dataset DA with iterative pruning based on Algorithm 2.
        """
        self.indicate()
        self.model.zero_grad()

        # Save initial parameters (θ0)
        initial_params = [w.clone().detach() for w in self.weights]

        # Compute initial gradients (g0) with num_batches
        initial_grads = [torch.zeros_like(w) for w in self.weights]
        batch_count = 0
        num_batches = max(num_batches, len(self.loader) / 32)
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            output = self.model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            grads = torch.autograd.grad(loss, self.weights, allow_unused=True)
            initial_grads = [g + (ag if ag is not None else torch.zeros_like(g)) 
                            for g, ag in zip(initial_grads, grads)]
            batch_count += 1
            if batch_count >= num_batches:
                break
        initial_grads = [g / batch_count for g in initial_grads]

        # Iterative pruning loop
        importance_scores = [torch.zeros_like(w) for w in self.weights]
        for i in range(num_iterations):
            ###
            current_sparsity = sparsity * (i + 1) / num_iterations
            self.model.train()
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                self.model.zero_grad()
                output = self.model(x)
                loss = torch.nn.CrossEntropyLoss()(output, y)
                loss.backward()
                with torch.no_grad():
                    for w in self.weights:
                        if w.grad is not None:
                            w.add_(-0.01 * w.grad)
                break
            masks = self.snip(current_sparsity, silent=True)
            for score, mask in zip(importance_scores, self.indicators):
                score += mask
            with torch.no_grad():
                for w, init_w in zip(self.weights, initial_params):
                    w.copy_(init_w)

        importance_scores = [score / num_iterations for score in importance_scores]
        data = {
            "theta_0": torch.cat([w.view(-1) for w in initial_params]).detach(),
            "g_0": torch.cat([g.view(-1) for g in initial_grads]).detach(),
            "importance": torch.cat([s.view(-1) for s in importance_scores]).detach()
        }
        torch.save(data, save_path)
        return data

def autos_prune(self, sparsity, autos_model_path, num_batches=10, silent=False):
    num_batches = max((len(self.loader) / 32), num_batches)
    
    # Loads the AutoS (MLP) model
    from utils.autosnet import MLP
    model = MLP().to(self.device)
    model.load_state_dict(torch.load(autos_model_path))
    model.eval()

    self.indicate()
    self.model.zero_grad()
    params = [w.view(-1).to(self.device) for w in self.weights]
    grads = [torch.zeros_like(w).view(-1) for w in self.weights]

    # Calculates the gradients of the original model from num_batches batches of data
    batch_count = 0
    for x, y in self.loader:
        x, y = x.to(self.device), y.to(self.device)
        self.model.zero_grad()
        x = self.model.forward(x)
        L = torch.nn.CrossEntropyLoss()(x, y)
        grads_batch = torch.autograd.grad(L, self.weights, allow_unused=True)
        grads = [
            g + (ag.view(-1) if ag is not None else torch.zeros_like(g.view(-1)))
            for g, ag in zip(grads, grads_batch)
        ]
        batch_count += 1
        if batch_count >= num_batches:
            break
    
    grads = [g / batch_count for g in grads]  # Average gradients
    params_tensor = torch.cat(params).to(self.device)
    grads_tensor = torch.cat(grads).to(self.device)
    # Debug: Print tensor shapes
    print(f"Params tensor shape: {params_tensor.shape}")
    print(f"Grads tensor shape: {grads_tensor.shape}")
    
    # Use DataLoader for batch processing
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(params_tensor, grads_tensor)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    importance_scores = []
    with torch.no_grad():
        for batch_params, batch_grads in loader:
            batch_params, batch_grads = batch_params.to(self.device), batch_grads.to(self.device)
            # Debug: Print batch shapes
            print(f"Batch shapes: params={batch_params.shape}, grads={batch_grads.shape}")
            
            output = model(batch_params, batch_grads).squeeze(-1)
            # Debug: Print output shape
            print(f"Output shape: {output.shape}")
            
            importance_scores.append(output.cpu())
    importance_scores = torch.cat(importance_scores)

    # Based on sparsity, it removes unimportant parameters
    thresh = float(importance_scores.kthvalue(int(sparsity * importance_scores.shape[0]))[0])
    idx = 0
    for j, layer in enumerate(self.indicators):
        layer_size = layer.numel()
        layer_scores = importance_scores[idx:idx + layer_size].view(layer.shape)
        layer[(layer_scores.abs() <= thresh)] = 0
        self.pruned[j] = int(torch.sum(layer == 0))
        idx += layer_size
    
    idx = 0
    # Update mask
    for name, param in self.mask_.named_parameters():
        if 'mask' not in name:
            param.data = self.indicators[idx]
            idx += 1
    
    if not silent:
        print("Weights left: ", [
            self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)
        ])
        print("Sparsities: ", [
            round(100 * pruned / self.indicators[i].numel(), 2) 
            for i, pruned in enumerate(self.pruned)
        ])
    
    return self.mask_
    
    def snipR(self, sparsity, silent=False):
        with torch.no_grad():
            saliences = [torch.zeros_like(w) for w in self.weights]
            x, y = next(iter(self.loader))
            z = self.model.forward(x)
            L0 = torch.nn.CrossEntropyLoss()(z, y)
            for laynum, layer in enumerate(self.weights):
                if not silent: print("layer ", laynum, "...")
                for weight in range(layer.numel()):
                    temp = layer.view(-1)[weight].clone()
                    layer.view(-1)[weight] = 0
                    z = self.model.forward(x)
                    L = torch.nn.CrossEntropyLoss()(z, y)
                    saliences[laynum].view(-1)[weight] = (L-L0).abs()    
                    layer.view(-1)[weight] = temp
                
            saliences_bag = torch.cat([s.view(-1) for s in saliences]).cpu()
            thresh = float(saliences_bag.kthvalue(int(sparsity * saliences_bag.numel()))[0])
            for j, layer in enumerate(self.indicators):
                layer[saliences[j] <= thresh] = 0
                self.pruned[j] = int(torch.sum(layer == 0))   
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel()-pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100*pruned/self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])

def cwi_importance(net, sparsity, device):
    mask = utils.net_utils.create_dense_mask_0(deepcopy(net), device, value=0)
    for (name, param), param_mask in zip(net.named_parameters(), mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            param_mask = abs(param.data) + abs(param.grad)
    imp = [layer for name, layer in mask.named_parameters() if 'mask' not in name]
    percentile = np.percentile(imp, sparsity*100)
    above_threshold = imp > percentile
    mask = imp * above_threshold
    return mask

def apply_reg(mask, model):
    for (name, param), param_mask in zip(model.named_parameters(), mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            l2_grad = param_mask.data * param.data
            param.grad += l2_grad

def update_reg(mask, reg_decay, cfg):
    reg_mask = create_dense_mask_0(deepcopy(mask), cfg.device, value=0)
    for (name, param), param_mask in zip(reg_mask.named_parameters(), mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            param.data[param_mask.data == 1] = 0
            if cfg.reg_type == 'x':
                if reg_decay < 1:
                    param.data[param_mask.data == 0] += min(reg_decay, 1)
            elif cfg.reg_type == 'x^2':
                if reg_decay < 1:
                    param.data[param_mask.data == 0] += min(reg_decay, 1)
                    param.data[param_mask.data == 0] = param.data[param_mask.data == 0]**2
            elif cfg.reg_type == 'x^3':
                if reg_decay < 1:
                    param.data[param_mask.data == 0] += min(reg_decay, 1)
                    param.data[param_mask.data == 0] = param.data[param_mask.data == 0]**3
    reg_decay += cfg.reg_granularity_prune
    return reg_mask, reg_decay
