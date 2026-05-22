import os, sys
import numpy as np
import torch
torch.cuda.init()
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg
import matplotlib.pyplot as plt
import wandb
import copy

from config import Config_MBM_fMRI
from dataset import hcp_dataset
from sc_mbm.mae_for_fmri import MAEforFMRI
from sc_mbm.trainer import train_one_epoch
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model
from torch.cuda.amp import GradScaler, autocast

'''
 这个模型不仅仅是在 fMRI 数据上训练 VAE/MAE，而是尝试将 fMRI 对齐到自然图像的表征空间，从而：
    (1)学习更具生物学意义的脑表征。
    (2)提高 fMRI 表征的泛化能力，避免仅仅是数据重建。

数据预处理
(1)加载 BOLD5000 数据（fMRI + 自然图像）。
(2)对 fMRI 进行随机掩蔽（类似 MAE）。
(3)ResNet50 提取自然图像的特征（冻结参数）。

训练 VAE/MAE
(1)fMRI -> 编码器 -> 潜在表示 -> 解码器 -> fMRI 重建。
(2)计算 fMRI 重建误差。

利用自然图像特征
(1)计算 fMRI 潜在表示 和 图像特征之间的距离。
(2)让 fMRI 学习更接近图像表征的潜在空间。
'''


os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

# 封装了Weights & Biases（wandb）库的功能来实现训练过程中的日志记录和可视化
# Weights & Biases（wandb）是一个机器学习实验管理工具，它主要用于在训练深度学习和其他机器学习模型时进行日志记录、可视化和结果追踪。
class wandb_logger:
    def __init__(self, config):
        wandb.init(
                    project="mind-vis",
                    anonymous="allow",
                    group='stageA_sc-mbm',
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    # 数据记录方法
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    # 监视模型的参数
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    # 记录图片
    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

## 添加保存和加载检查点的函数
def save_checkpoint(state, filename):
    """保存检查点到硬盘"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """加载检查点，如果存在的话"""
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

## 添加创建和验证检查点目录的函数
def ensure_checkpoint_directory(base_path):
    """确保检查点目录存在，如果不存在，则创建它"""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    print(f"Checkpoint directory created at: {base_path}")
    return base_path


def get_args_parser():
    parser = argparse.ArgumentParser('MBM pre-training for fMRI', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    parser.add_argument('--mask_ratio', type=float) # 掩码比率，表示 fMRI 数据在训练时被随机屏蔽的比例。
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--num_heads', type=int) # 编码器的多头注意力头数，影响 Transformer 计算复杂度和建模能力
    parser.add_argument('--decoder_num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float) # MLP 扩展比，决定 MLP 层的隐藏单元数，相对于 embed_dim 的倍数

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)

    parser.add_argument('--include_hcp', type=bool)
    parser.add_argument('--include_kam', type=bool)

    parser.add_argument('--use_nature_img_loss', type=bool)
    parser.add_argument('--img_recon_weight', type=float)
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
                        
    return parser

# 创建一个readMe的文件
def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

# fmri数据稀疏化
def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels 形状为 (1, num_voxels) 的数组，它通常代表了一个单一时间点上所有体素（voxels）的活动强度。
    # int(x.shape[0]*sparse_rate)个体素被选中, 置0
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(config):

    # 碎片化检查
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:24'

    # 定义检查点存储的基本路径
    base_checkpoint_dir = os.path.join(config.output_path, 'checkpoints')
    # 为每个训练会话创建一个带时间戳的特定目录
    session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = ensure_checkpoint_directory(os.path.join(base_checkpoint_dir, session_id))
    
    # 检查点文件路径
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank)
        # 多节点的情况下分布式计算 
        torch.distributed.init_process_group(backend='nccl')
    # 创建一个多时间戳的实验结果目录
    output_path = os.path.join(config.root_path, 'results', 'fmri_pretrain',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    # output_path = os.path.join(config.root_path, 'results', 'fmri_pretrain')
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
    
    # 初始化日志服务器的后端
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    # 设置随机种子数, 随机数的作用就是对齐每一次的实验结果
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create dataset and dataloader
    dataset_pretrain = hcp_dataset(path=os.path.join(config.root_path, 'data/HCP/npz'), roi=config.roi, patch_size=config.patch_size,
                transform=fmri_transform, aug_times=config.aug_times, num_sub_limit=config.num_sub_limit, 
                include_kam=config.include_kam, include_hcp=config.include_hcp)
   
    print(f'Dataset size: {len(dataset_pretrain)}\nNumber of voxels: {dataset_pretrain.num_voxels}')
    # 创建分布式采样器（反正我只有一块）
    sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    dataloader_hcp = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, 
                shuffle=(sampler is None), pin_memory=True)

    # create model 创建模型，创建一个掩蔽的模型
    config.num_voxels = dataset_pretrain.num_voxels
    model = MAEforFMRI(num_voxels=dataset_pretrain.num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                    decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
                    num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                    focus_range=config.focus_range, focus_rate=config.focus_rate, 
                    img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss)   
    model.to(device)
    model_without_ddp = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

    # 参数的优化
    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    start_epoch = 0

    # 尝试加载检查点
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    cor_list = []
    start_time = time.time()
    print('Start Training the fmri MAE ... ...')

    print(config.batch_size)

    img_feature_extractor = None
    preprocess = None
    if config.use_nature_img_loss:
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.models.feature_extraction import create_feature_extractor
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        m = resnet50(weights=weights)   
        # 在这里使用resnet50用于提取图像的特征，并且在冻结参数的情况下
        img_feature_extractor = create_feature_extractor(m, return_nodes={f'layer2': 'layer2'}).to(device).eval()
        # 这里是冻结参数，保证参数不会进行更新
        for param in img_feature_extractor.parameters():
            param.requires_grad = False

    try:
        for ep in range(start_epoch,config.num_epoch):
            if torch.cuda.device_count() > 1: 
                sampler.set_epoch(ep) # to shuffle the data at every epoch

            # 计算当前训练轮数的相关系数
            cor = train_one_epoch(model, dataloader_hcp, optimizer, device, ep, loss_scaler, logger, config, start_time, model_without_ddp,
                                img_feature_extractor, preprocess)
            cor_list.append(cor)
            # 每20轮保存一次图像，并且重构一下fMRI图像
            if (ep % 20 == 0 or ep + 1 == config.num_epoch) and ep != 0 and config.local_rank == 0:
                # save models
                save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
                # plot figures
                plot_recon_figures(model, device, dataset_pretrain, output_path, 5, config, logger, model_without_ddp)
                
            # 每个 epoch 结束后保存检查点
            save_checkpoint({
                'epoch': ep + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_path)

            # 在这里也执行一次内存清理
            if ep % 20 == 0 or ep + 1 == config.num_epoch:
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"Exception occurred: {e}, saving checkpoint at epoch {ep}")
        save_checkpoint({
            'epoch': ep,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
    }, filename=checkpoint_path)
        raise
       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:  
        logger.log('max cor', np.max(cor_list), step=config.num_epoch-1)
        logger.finish()
    return

@torch.no_grad()
# 可视化模型的重构的能力
def plot_recon_figures(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['fmri']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        sample_with_mask = model_without_ddp.patchify(sample).to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').numpy().reshape(-1)
        sample = sample.to('cpu').numpy().reshape(-1)
        mask = mask.to('cpu').numpy().reshape(-1)
        # cal the cor
        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_fMRI()
    config = update_config(args, config)
    main(config)
    