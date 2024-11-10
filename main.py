# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import math
import random
import numpy as np
from PIL import Image

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS, AdversarialNetwork
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.transform import get_transform
from utils.utils import visda_acc, get_mask, re_org_img, mix_img

from torchvision import transforms, datasets
from data.data_list_image import ImageList, ImageListIndex, rgb_loader
from models import lossZoo
import networkx as nx
from models.TVTmodeling import TVTVisionTransformer, TVTAdversarialNetwork
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def postprocess_activations(activations):
    output = activations
    output *= 255
    #return output.astype('uint8')
    return 255 - output.astype('uint8')

def apply_heatmap(weights, img):
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
    return heatmap
    
def get_heatmaps(activations, img):
    weights = postprocess_activations(activations)
    heatmap = apply_heatmap(weights, img)
    return heatmap

def apply_heatmap_ori(weights, img):
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0, img, 1, 0)
    return heatmap
    
def get_ori(activations, img):
    weights = postprocess_activations(activations)
    heatmap = apply_heatmap_ori(weights, img)
    return heatmap

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, prefix_saved_mode, is_adv=False, ):
    model_to_save = model.module if hasattr(model, 'module') else model
    if is_adv:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, prefix_saved_mode+args.name+"_checkpoint_adv"+"_"+".bin" )
    else:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, prefix_saved_mode+args.name+"_checkpoint"+"_"+".bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))

def setup(args, prefix_saved_mode):
    # Prepare model
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, 
                              num_classes=args.num_classes, msa_layer=args.msa_layer)
    
    best_acc = 0
    best_model = None
    if(not os.path.exists('./output/'+args.dataset)):
        os.makedirs('./output/'+args.dataset)
    #print('prefix_saved_mode ',prefix_saved_mode)
    for file in os.listdir('./output/'+args.dataset):
        #print('file', file)
        if(prefix_saved_mode in file and 'checkpoint' in file ):
            #print('file', file)
            if(best_acc > float(file.split('_')[4] ) ):
                os.remove('./output/'+args.dataset+'/'+file)

    if(best_model is not None):
        model_checkpoint = os.path.join(args.output_dir, args.dataset, best_model)
        model.load_state_dict(torch.load(model_checkpoint))
    else:
        print('pretrained model:', args.pretrained_dir)
        model.load_from(np.load(args.pretrained_dir))
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step, cp_mask, ad_net, prefix_saved_mode):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    ad_net.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    valid_cp = 0 
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():
            logits, _, _, temp_cp = model(x_s = x, ad_net=ad_net, cp_mask=cp_mask, \
                 optimal_flag = args.optimal, )
            valid_cp = valid_cp + temp_cp.detach()

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
    #plt.imsave('./output/'+args.dataset+'/'+prefix_saved_mode+'valid_cp_mask.jpeg',valid_cp.to('cpu'), cmap='rainbow' )  
    #np.savetxt('./output/'+args.dataset+'/'+prefix_saved_mode+'valid_cp_mask.csv', np.array(valid_cp.to('cpu')) )

    all_preds, all_label = all_preds[0], all_label[0]
    if args.dataset == 'visda17':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results of: %s" % args.name)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    if args.dataset == 'visda17':
        return accuracy, classWise_acc
    else:
        return accuracy, None


def train(args, model,cp_mask, prefix_saved_mode):
    best_acc = 0
    for file in os.listdir('./output/'+args.dataset):
        if(prefix_saved_mode in file and 'checkpoint' in file ):
            if(best_acc < float(file.split('_')[4] ) ):
                best_acc = float(file.split('_')[4])

    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)
    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.source_list).readlines(), transform=transform_source, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    
    target_loader = torch.utils.data.DataLoader(
        ImageListIndex(open(args.target_list).readlines(), transform=transform_target, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
    
    config = CONFIGS[args.model_type]
    ad_net = AdversarialNetwork(config.hidden_size, config.hidden_size//4)
    ad_net.to(args.device)
    ad_net_local = AdversarialNetwork(config.hidden_size//12, config.hidden_size//12)
    ad_net_local.to(args.device)
    
    optimizer_ad = torch.optim.SGD(list(ad_net.parameters())+list(ad_net_local.parameters()),
                            lr=args.learning_rate/10, 
                            momentum=0.9,
                            weight_decay=args.weight_decay)
    
    optimizer = torch.optim.SGD([
                                    {'params': model.transformer.parameters(), 'lr': args.learning_rate/10},
                                    {'params': model.decoder.parameters(), 'lr': args.learning_rate},
                                    {'params': model.head.parameters()},
                                ],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
        
    model.zero_grad()
    ad_net.zero_grad()
    ad_net_local.zero_grad()
        
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    best_classWise_acc = ''

    len_source = len(source_loader)
    len_target = len(target_loader)            

    for global_step in range(1, t_total):
        model.train()
        ad_net.train()
        ad_net_local.train()

        if (global_step-1) % (len_source-1) == 0:
            iter_source = iter(source_loader)    
        if (global_step-1) % (len_target-1) == 0:
            iter_target = iter(target_loader)
        
        data_source = next(iter_source)
        data_target = next(iter_target)

        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        x_t, _, index_t = tuple(t.to(args.device) for t in data_target)
        
        if( not args.use_cp ):
            cp_mask = np.ones( (257, 257))
            cp_mask = torch.from_numpy(cp_mask).float().to(args.device)

        logits_s, logits_t, loss_ad_local, loss_rec, x_s, x_t, temp_mask = model(x_s = x_s, x_t = x_t, ad_net = ad_net_local, cp_mask=cp_mask, \
            optimal_flag = args.optimal, )
        cp_mask = temp_mask
        plt.imsave('./output/'+args.dataset+'/'+prefix_saved_mode+'train_cp_mask.jpeg',cp_mask[1:,1:].to('cpu'), cmap='rainbow' )  
        np.savetxt('./output/'+args.dataset+'/'+prefix_saved_mode+'train_cp_mask.csv', np.array(cp_mask[1:,1:].to('cpu')) )

        loss_fct = CrossEntropyLoss()
        loss_clc = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1))
        
        loss_im = lossZoo.im(logits_t.view(-1, args.num_classes))
        loss_ad_global = lossZoo.adv(torch.cat((x_s[:,0], x_t[:,0]), 0), ad_net)
        loss = loss_clc + args.beta * loss_ad_global + args.gamma * loss_ad_local

        if args.use_im:
            loss += (args.theta * loss_im)
        #print('loss_clc {}, loss_im {}, loss_ad_global {}, loss_ad_local {}'.format(loss_clc.detach(), loss_im.detach(), loss_ad_global.detach(), loss_ad_local.detach()))
            
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_local.parameters(), args.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        optimizer_ad.step()
        optimizer_ad.zero_grad()
        scheduler_ad.step()
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_clc", scalar_value=loss_clc.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_global", scalar_value=loss_ad_global.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_local", scalar_value=loss_ad_local.item(), global_step=global_step)
            writer.add_scalar("train/loss_rec", scalar_value=loss_rec.item(), global_step=global_step)
            writer.add_scalar("train/loss_im", scalar_value=loss_im.item(), global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
        
        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
            accuracy, classWise_acc = valid(args, model, writer, test_loader, global_step, cp_mask, ad_net_local, prefix_saved_mode)
            if best_acc < accuracy:
                best_acc = accuracy

                save_model(args, model, prefix_saved_mode +str(best_acc) +'_',  is_adv=False, )
                save_model(args, ad_net_local, prefix_saved_mode +str(best_acc) +'_', is_adv=True, )
                for file in os.listdir('./output/'+args.dataset):
                    if(prefix_saved_mode in file and 'checkpoint' in file ):
                        if(best_acc > float(file.split('_')[4] ) ):
                            os.remove('./output/'+args.dataset+'/'+file)

                if classWise_acc is not None:
                    best_classWise_acc = classWise_acc
            model.train()
            ad_net_local.train()
            logger.info("Current Best Accuracy: %2.5f" % best_acc)
            logger.info("Current Best element-wise acc: %s" % best_classWise_acc)
        
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Best element-wise Accuracy: \t%s" % best_classWise_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cp", default=False, action="store_true",
                        help="Use Core periphery constraint.")

    parser.add_argument("--name", default = 'qs', type=str, 
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default = 'DomainNet', type=str, help="Which downstream task.")
    parser.add_argument("--source_list", default = 'data/DomainNet/quickdraw_train.txt', type=str, help="Path of the training data.")
    parser.add_argument("--target_list", default = 'data/DomainNet/sketch_train.txt', type=str, help="Path of the test data.")
    parser.add_argument("--test_list", default = 'data/DomainNet/sketch_test.txt', type=str, help="Path of the test data.")
    parser.add_argument("--num_classes", default=345, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/imagenet21k_ViT-B_16.npz", #imagenet21k_ViT-B_16.npz  ViT-B_16.npz
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=256, type=int, #256
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--beta", default=0.1, type=float,   #alpha 
                        help="The importance of the adversarial loss.")
    parser.add_argument("--gamma", default=0.1, type=float,  #beta
                        help="The importance of the local adversarial loss.")
    parser.add_argument("--theta", default=0.1, type=float,  #gamma
                        help="The importance of the IM loss.")
    parser.add_argument("--use_im", default=False, action="store_true",
                        help="Use information maximization loss.")
    parser.add_argument("--msa_layer", default=12, type=int,
                        help="The layer that incorporates local alignment.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")

    parser.add_argument("--learning_rate", default=0.05, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=5000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--gpu_id', default='1', type = str,
                        help="gpu id")
    parser.add_argument('--optimal', default=0, type = int,
                        help="use optimal linear transform noise, 1 means use, 0 means no use")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cuda = True if torch.cuda.is_available() else False
    if torch.cuda.device_count() > 0:
       print("Let's use", torch.cuda.device_count(), "GPUs!")
    #device = torch.device('cuda' if cuda else 'cpu')

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    
    set_seed(args)


    if( not args.use_cp ):
        prefix_saved_mode = args.name+'_NoCP_'+'Perturbation_' + str(args.optimal) + '_'
    else:
        prefix_saved_mode = args.name+'_SoftCP_'+'Perturbation_' + str(args.optimal) + '_'

    num_patchs = int(args.img_size * args.img_size / 16 / 16)
    cp_mask = np.ones( (num_patchs+1, num_patchs+1))
    cp_mask = torch.from_numpy(cp_mask).float().to(args.device)
    
    args, model = setup(args, prefix_saved_mode)
    model.to(args.device)
    train(args, model,cp_mask, prefix_saved_mode)


if __name__ == "__main__":
    main()
