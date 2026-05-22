import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wandb
import copy

# own code
from config import Config_MBM_finetune
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from sc_mbm.mae_for_fmri import MAEforFMRI
from sc_mbm.trainer import train_one_epoch, train_one_epoch_ft
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model


os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

'''
stageA2_mbm_finetune.py是一个针对特定fMRI数据集进行模型微调的脚本
'''
# 这部分还是使用wandb_logger记录训练的信息
'''
要使用wandb_logger和查看日志，你需要有一个Weights & Biases（W&B）账户。首先，在W&B网站注册并登录。
然后，在代码中使用wandb_logger时，会自动将训练过程的日志数据发送到W&B。
你可以在W&B的项目仪表板上查看这些数据，包括指标图表、图像和其他可视化结果。
具体步骤包括初始化wandb_logger、在训练循环中记录所需的数据和图像，以及训练结束后查看W&B仪表板上的项目日志。
'''
class wandb_logger:
    def __init__(self, config):
        wandb.init( project='mind-vis',
                    group="stepA_sc-mbm_tune",
                    anonymous="allow",
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

# 相关的参数
def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning on Test fMRI', add_help=False)

    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mask_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--pretrain_mbm_path', 
                        default='/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_pretrain/All_rois/checkpoints/checkpoint.pth',
                        type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--include_nonavg_test', type=bool)   
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
                        
    return parser

# 创建一个包含配置信息的readme文件
def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

# 请你对fMRI数据进行变换，并且选择一定的比例的体素数据置0以后增加数据的系数的稀疏性
def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def sliding_average(data, window_size):
    # 确保数据是2D的，以便应用滑动平均
    data = np.atleast_2d(data)

    # 定义卷积核，这里使用平均滤波器
    kernel = np.ones(window_size) / window_size

    # 使用convolve1d进行滑动平均
    averaged_data = convolve1d(data, kernel, axis=1, mode='constant', cval=0.0)

    return averaged_data

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')


def main(config, subdir):
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    
    # 这里加载预训练模型
    # sd = torch.load(config.pretrain_mbm_path, map_location='cpu')
    config.pretrain_mbm_path = f'/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_pretrain/{subdir}/checkpoints/checkpoint.pth'
    sd = torch.load(config.pretrain_mbm_path, map_location='cpu')
    config_pretrain = sd['config']
    
    # 模型保存目录
    # subdir = os.path.basename(os.path.dirname(os.path.dirname(config.pretrain_mbm_path)))
    # subdir = 'All_rois'
    output_path = os.path.join(
        config.root_path,
        'results',
        'fmri_finetune',
        subdir,
        datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    )
    config.output_path = output_path


    # output_path = os.path.join(config.root_path, 'results', 'fmri_finetune',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    # output_path = os.path.join(config.root_path, 'results', 'fmri_finetune')
    # config.output_path = output_path

    logger = wandb_logger(config) if config.local_rank == 0 else None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config_pretrain.seed)
    np.random.seed(config_pretrain.seed)

    # # # 这部分数据是可以选择不同的脑区的
    # roi = ['PHA1', 'PHA2', 'PHA3', 'TPOJ2', 'TPOJ3', 'DVT', 'PGp', 'IP0', 'PGi']
    # roi = ['V1', 'V2', 'V3', 'V4',
    #        'MST', 'V8', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'MT', 'PCV', 'POS1', 'MIP',
    #        'LIPd', 'VMV3', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', 'VVC', 'PHT', 'PH',
    #        'PHA1', 'PHA2', 'PHA3', 'TPOJ2', 'TPOJ3', 'DVT', 'PGp', 'IP0', 'PGi']
    
    # subj = 1
    # fmri_tr = []
    # fmri_te = []
    # for croi in roi:
    #     cX = np.load(f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/mrifeat_new/subj0{subj}/subj0{subj}_{croi}_betas_ave_tr.npy').astype("float32")
    #     cX_te = np.load(f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/mrifeat_new/subj0{subj}/subj0{subj}_{croi}_betas_ave_te.npy').astype("float32")
    #     fmri_tr.append(cX)
    #     fmri_te.append(cX_te)
    # train_set = np.hstack(fmri_tr)
    # test_set = np.hstack(fmri_te)
    
    # 加载fmri数据：测试集和训练集,这部分使用了所有34个脑区的数据
    subjects=[1]
    # subjects = [1]
    train_sets, test_sets = [], []
    for subj in subjects:
        train_data = np.load(
        f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/mrifeat_new/subj0{subj}/subj0{subj}_vis_rois_betas_ave_tr.npy')
        
        train_sets.append(train_data)
    train_set = np.concatenate(train_sets, axis=0)
    for subj in subjects:
        test_data = np.load(
        f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/mrifeat_new/subj0{subj}/subj0{subj}_vis_rois_betas_ave_te.npy')
        test_sets.append(test_data)
    test_set = np.concatenate(test_sets, axis=0)

    train_set = train_set / 300
    test_set = test_set / 300

    train_set = pad_to_patch_size(train_set, 16)
    test_set = pad_to_patch_size(test_set, 16)

    train_set = np.vstack((train_set,test_set))
    print(train_set.shape)

    # train_set = np.concatenate((train_set,test_set), axis=0)
    print(f'train_set shape:{train_set.shape}')

    # create model
    num_voxels = (sd['model']['pos_embed'].shape[1] - 1) * config_pretrain.patch_size
    model = MAEforFMRI(num_voxels=num_voxels, patch_size=config_pretrain.patch_size,
                       embed_dim=config_pretrain.embed_dim,
                       decoder_embed_dim=config_pretrain.decoder_embed_dim, depth=config_pretrain.depth,
                       num_heads=config_pretrain.num_heads, decoder_num_heads=config_pretrain.decoder_num_heads,
                       mlp_ratio=config_pretrain.mlp_ratio, focus_range=None, use_nature_img_loss=False)
    # 加载当前的模型的状态字典到当前的模型
    model.load_state_dict(sd['model'], strict=False)

    # 冻结 patch_embed 和前 N-3 个 Transformer block，只微调最后 3 个 block
    for param in model.patch_embed.parameters():
        param.requires_grad = False

    for blk in model.blocks:
        for param in blk.parameters():
            param.requires_grad = False

    num_blocks_to_train = 3
    for blk in model.blocks[-num_blocks_to_train:]:
        for param in blk.parameters():
            param.requires_grad = True

    model.to(device)
    model_without_ddp = model

    sampler = torch.utils.data.DistributedSampler(train_set) if torch.cuda.device_count() > 1 else torch.utils.data.RandomSampler(train_set)
    dataloader_hcp = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

    # 推荐用于 transformer 微调
    #param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    #optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    # 虽然只优化可训练参数，但没有区分是否需要 weight decay，容易导致 LayerNorm 出问题（训练不稳定）
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, betas=(0.9, 0.95))

    # 你可以配合打印一下参与训练的参数：
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    cor_list = []
    start_time = time.time()
    print('Finetuning MAE on test fMRI ... ...')
    # 在预训练模型的基础上，使用上述的特地给数据集进行微调，以方便适应特征数据集的权重，改善其数据的表现
    for ep in range(30):
        epoch_save_dir = f'epoch_{ep}'
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(ep)  # to shuffle the data at every epoch
        cor = train_one_epoch_ft(model, dataloader_hcp, optimizer, device, ep, loss_scaler, logger, config, start_time,
                              model_without_ddp)
        cor_list.append(cor)
        if (ep % 5 == 0 or ep == 19) and ep != 0  and config.local_rank == 0:
            # save models
            save_model(config_pretrain, ep, model_without_ddp, optimizer, loss_scaler,
                       os.path.join(output_path, epoch_save_dir, 'checkpoints'))
            # plot figures
            plot_recon_figures_ft(model, device, test_set, os.path.join(output_path, epoch_save_dir),ep , 5,logger, model_without_ddp)
  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        logger.log('max cor', np.max(cor_list), step=config.num_epoch-1)
        logger.finish()
    return

# 预训练的时候使用
@torch.no_grad()
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

# 微调的时候使用
@torch.no_grad()
def plot_recon_figures_ft(model, device, dataset, output_path, ep, num_figures=5, logger=None,
                          model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    save_dir = os.path.join(output_path, f'epoch_{ep}_images')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_figures):
        sample = next(iter(dataloader))
        sample = sample.reshape(sample.shape[0], 1, -1)
        sample = sample.float().to(device)
        _, _, pred, mask = model(sample, mask_ratio=0.75)
        # print(latent.shape)
        sample_with_mask = model_without_ddp.patchify(sample).to('cpu').numpy().reshape(-1,
                                                                                        model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).detach().to('cpu').numpy().reshape(-1)
        sample = sample.to('cpu').numpy().reshape(-1)
        mask = mask.to('cpu').numpy().reshape(-1)
        # cal the cor
        cor = np.corrcoef([pred, sample])[0, 1]

        x_axis = np.arange(0, sample.shape[-1])

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_title('Ground-truth and Reconstruction')
        # groundtruth
        ax.plot(x_axis, sample, label='Ground-truth')
        # pred
        ax.plot(x_axis, pred, label='Reconstruction')
        ax.set_ylabel('cor: %.4f' % cor, weight='bold')
        ax.yaxis.set_label_position("right")
        ax.legend()  # 添加图例

        image_path = os.path.join(save_dir, f'reconstruction_{i}.png')
        plt.savefig(image_path)
        if logger is not None:
            logger.log_image('reconst', fig)
        plt.close()  # 关闭当前图，确保下一次循环创建新图



def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

if __name__ == '__main__':



    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_finetune()
    config = update_config(args, config)

    subdirs = ['LVC'] # 'V1-V4_rois',  All_rois, OnlyVision, Other_rois, noVC
    for subdir in subdirs:
        # 开始时间
        start_time = time.time()  # 记录开始时间
        main(config, subdir)
        #结束时间
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 单位是秒
        elapsed_minutes = elapsed_time / 60
        print(f"Total running time: {elapsed_minutes:.2f} minutes")

    '''
    only vision就是除了other
    novc就是other
    lvc就是v1-v4
    '''
