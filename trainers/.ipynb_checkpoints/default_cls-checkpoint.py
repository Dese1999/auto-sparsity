import time
import torch
import numpy as np
import torch.nn as nn
from utils import net_utils
from layers.CS_KD import KDLoss
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.pruning import apply_reg, update_reg
import matplotlib.pyplot as plt
from models.split_resnet import binarize


__all__ = ["train", "validate"]



kdloss = KDLoss(4).cuda()

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):        
        m.eval()

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()
                
def train(train_loader, model, criterion, optimizer, epoch, cfg, writer, mask=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5], cfg,
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()

    
    n_total_params = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n and 'bn' not in n and 'downsample' not in n)
    for i, data in enumerate(train_loader):
        images, target = data[0].cuda(), data[1].long().squeeze().cuda()
        data_time.update(time.time() - end)

        if cfg.cs_kd:
            batch_size = images.size(0)
            loss_batch_size = batch_size // 2
            targets_ = target[:batch_size // 2]
            outputs = model(images[:batch_size // 2])
            loss = torch.mean(criterion(outputs, targets_))
            with torch.no_grad():
                outputs_cls = model(images[batch_size // 2:])
            cls_loss = kdloss(outputs[:batch_size // 2], outputs_cls.detach())
            lamda = 3
            loss += lamda * cls_loss
            acc1, acc5 = accuracy(outputs, targets_, topk=(1, 5))
        else:
            batch_size = images.size(0)
            loss_batch_size = batch_size
            output = model(images)
            if cfg.use_noisy_logit:
                output = output + torch.normal(mean=0, std=1, size=(output.shape[0], output.shape[1]))
            loss = criterion(output, target)

            # Sparsity penalty for learnable masks
            bin_all = 0
            neg_bin = 0
            for idx, m in enumerate(model.binary_masks):
                bin_mask = binarize(m)
                zeros = torch.sum(bin_mask == 0).item()
                total = m.numel()
                bin_count = torch.sum(bin_mask).item()
                neg_count = total - bin_count
                bin_all += bin_count
                neg_bin += neg_count
                #print(f"Layer {idx} sparsity: {zeros}/{total} zeros ({zeros/total*100:.2f}%)")

            bin_percent = bin_all / n_total_params if n_total_params > 0 else 0
            neg_percent = neg_bin / n_total_params if n_total_params > 0 else 0
            sparsity_penalty = cfg.lamda_sparse * bin_percent + cfg.lamda_sparse * cfg.n_ratio * neg_percent
            sparsity_loss = cfg.lamda_sparse * (neg_percent - cfg.sparsity) ** 2
            loss += sparsity_penalty + sparsity_loss

            # Binarization penalty
            binary_loss = 0
            for m in model.binary_masks:
                binary_loss += torch.mean((m - (m > 0).float()) ** 2)
            loss += cfg.lambda_binary * binary_loss
            writer.add_scalar("train/binary_loss", binary_loss.item(), (num_batches * epoch + i) * batch_size)

            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), loss_batch_size)
        top1.update(acc1.item(), loss_batch_size)
        top5.update(acc5.item(), loss_batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0 or i == num_batches - 1:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)
            writer.add_scalar("train/sparse_percent", neg_percent, t)

    return top1.avg, top5.avg, neg_percent

#################################################################################    
def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=True)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5],args, prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        # confusion_matrix = torch.zeros(args.num_cls,args.num_cls)
        for i, data in enumerate(val_loader):
            # images, target = data[0]['data'], data[0]['label'].long().squeeze()
            images, target = data[0].cuda(), data[1].long().squeeze().cuda()

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # print(target,torch.mean(images),acc1,acc5,loss,torch.mean(output))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)
        # if epoch%10==0:

        print(top1.avg, top5.avg )
    return top1.avg, top5.avg
