import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from scipy.ndimage import convolve1d
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy
from sklearn.preprocessing import MaxAbsScaler

# own code
from config import Config_MBM_finetune, Config_MBM_fMRI
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from sc_mbm.mae_for_fmri import MAEforFMRI, fMRI2CLIP, fMRI2Latent
from sc_mbm.trainer_latent import train_one_epoch_feat,validate, train_one_epoch_feat_rdm
from sc_mbm.trainer_latent import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."
os.environ['WANDB_MODE'] = "offline"

ROI_NAME = 'all'

# if ROI_NAME == 'lowvis':
#     DATA_ROI_NAME = 'LVC/epoch_99'
# elif ROI_NAME == 'all':
#     DATA_ROI_NAME = 'All_rois/27-05-2025-15-49-11/epoch_40'
# elif ROI_NAME == 'visual':
#     DATA_ROI_NAME = 'OnlyVision/epoch55'
# elif ROI_NAME == 'other':
#     DATA_ROI_NAME = 'noVC/epoch55'


def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

class wandb_logger:
    def __init__(self, config):
        wandb.init(project='Pre2Clip_Text',
                   group='Pre2Clip',
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


def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning on Test fMRI', add_help=False)

    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mask_ratio', type=float)

    # Project setting
    parser.add_argument("--subject", type=str, default='subj01')
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--pretrain_mbm_path',
                        # default=f'/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_finetune_save/{DATA_ROI_NAME}/checkpoints/checkpoint.pth',
                        # default='/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_finetune_save/fmri_data/All_rois/29-06-2025-17-07-13/epoch_65/checkpoints/checkpoint.pth',
                        default = '/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_finetune_save/fmri_data/All_rois/subj01/share/epoch_40/checkpoints/checkpoint.pth',
                        type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--include_nonavg_test', type=bool)

    # distributed training parameters
    parser.add_argument('--local_rank', type=int)

    return parser


def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0] * sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

class FMRIAndFeatDataset(Dataset):
    def __init__(self, fmri_data, feat_data):
        self.fmri_data = fmri_data
        self.feat_data = feat_data

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri_tensor = torch.tensor(self.fmri_data[idx], dtype=torch.float32)
        feat_tensor = torch.tensor(self.feat_data[idx], dtype=torch.float32)

        return fmri_tensor, feat_tensor

def get_dataloader(fmri_data, feat_data, batch_size=1, shuffle=True, num_workers=1):
    dataset = FMRIAndFeatDataset(fmri_data, feat_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def main(config):
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    sd = torch.load(config.pretrain_mbm_path, map_location='cpu')
    config_pretrain = sd['config']

    output_path = os.path.join(config.root_path, 'results', 'fmri_2_clip',
                               '%s_%s' % (config.subject, datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    # output_path = os.path.join(config.root_path, 'results', 'fmri_finetune')
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
 
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)

    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config_pretrain.seed)
    np.random.seed(config_pretrain.seed)


    # 加载某一被试的训练/测试 fMRI 信号数据（形状通常是 [样本数, voxel数]）
    fmri_tr = []
    fmri_te = []
    testsubj=config.subject
    print("test subj: ", testsubj)
    target = 'vdv_latent' # init_latent c
    print("target: ", target)
    path_root = '/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main'

    # subjects = [1]
    # train_sets = []
    # for subj in subjects:
    #     train_data = np.load(f'{path_root}/mrifeat_0526/subj0{subj}/subj0{subj}_all_betas_ave_tr.npy')
    #     train_sets.append(train_data)
    #     fmri_tr = np.concatenate(train_sets, axis=0)
   
    fmri_tr = np.load(f'{path_root}/mrifeat/{testsubj}/{testsubj}_{ROI_NAME}_rois_betas_ave_tr_aligned.npy')
    fmri_te = np.load(f'{path_root}/mrifeat/{testsubj}/{testsubj}_{ROI_NAME}_rois_betas_ave_te_aligned.npy')
    # fmri_tr = np.load(f'/home/dell/Codes/StableDiffusionReconstruction/mrifeat_new/{test}/{test}_all_rois_betas_ave_tr.npy')
    # fmri_te = np.load(f'/home/dell/Codes/StableDiffusionReconstruction/mrifeat_new/{test}/{test}_all_rois_betas_ave_te.npy')

    fmri_tr = fmri_tr / 300
    fmri_te = fmri_te / 300

    fmri_tr = pad_to_patch_size(fmri_tr, 16)
    fmri_te = pad_to_patch_size(fmri_te, 16)

    # 加载对应的 CLIP 特征
    if target == 'c':
        CLIP_tr = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_tr.npy').reshape(-1, 77, 768)
        CLIP_te = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_te.npy').reshape(-1, 77, 768)
        train_dataloader = get_dataloader(fmri_tr, CLIP_tr, batch_size=config.batch_size, shuffle=True)
        test_dataloader = get_dataloader(fmri_te, CLIP_te, batch_size=config.batch_size, shuffle=False)

    elif target == 'vdv_latent':
        # 将 latent 数据重塑为合适的形状用于 fMRI2Latent
        init_latent_tr = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_tr.npy')
        init_latent_te = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_te.npy')
        
        # 原始形状: (samples, 257, 768)
        print(f"Original latent shape: {init_latent_tr.shape}")
        
        # 使用完整的257*768维度，通过降低batch_size和使用梯度累积来处理
        # 保持完整维度，确保后续图像生成时有足够的信息
        init_latent_tr = init_latent_tr.reshape(init_latent_tr.shape[0], -1)  # (samples, 197376)
        init_latent_te = init_latent_te.reshape(init_latent_te.shape[0], -1)  # (samples, 197376)
        
        print(f"Full latent shape: {init_latent_tr.shape}")
        print(f"Latent dimension: {init_latent_tr.shape[1]}")
        
        # 使用更小的batch_size来适应大维度，并增加梯度累积
        train_dataloader = get_dataloader(fmri_tr, init_latent_tr, batch_size=1, shuffle=True)  # 从 4 降低到 1
        test_dataloader = get_dataloader(fmri_te, init_latent_te, batch_size=1, shuffle=False)

    # train_dataloader = get_dataloader(fmri_tr, init_latent_tr, batch_size=32, shuffle=True)
    # test_dataloader = get_dataloader(fmri_te, init_latent_te, batch_size=32, shuffle=False)

    # train_set = np.concatenate((train_set,test_set), axis=0)


    # create modelprint(f'train_set shape:{fmri_tr.shape}')
    print(f'fMRI training data shape: {fmri_tr.shape}')
    print(f'fMRI test data shape: {fmri_te.shape}')
    
    num_voxels = (sd['model']['pos_embed'].shape[1] - 1) * config_pretrain.patch_size
    print(f'Model expects num_voxels: {num_voxels}')
    print(f'Actual fMRI voxels: {fmri_tr.shape[1]}')
    
    # 检查维度匹配
    if fmri_tr.shape[1] != num_voxels:
        print(f"Warning: Dimension mismatch! Model expects {num_voxels}, but data has {fmri_tr.shape[1]} voxels")
    
    # 确定目标输出维度 - 使用完整的257*768维度
    target_output_dim = 257 * 768  # 197376，保持完整维度用于后续图像生成
    print(f"Target output dimension: {target_output_dim}")
    
    # 创建模型
    model = fMRI2Latent(num_voxels=num_voxels, 
                    patch_size=config_pretrain.patch_size,
                    embed_dim=config_pretrain.embed_dim, 
                    decoder_depth=config.decoder_depth,  # 使用当前finetune配置的decoder_depth
                    decoder_embed_dim=config_pretrain.decoder_embed_dim, 
                    depth=config_pretrain.depth,
                    num_heads=config_pretrain.num_heads, 
                    decoder_num_heads=config_pretrain.decoder_num_heads,
                    mlp_ratio=config_pretrain.mlp_ratio, 
                    focus_range=None, 
                    use_nature_img_loss=False,
                    output_dim=target_output_dim)  # 新参数：输出维度
    
    # 智能加载预训练权重
    model_dict = model.state_dict()
    pretrained_dict = sd['model']
    
    # 过滤掉维度不匹配的键
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f"Loaded {len(pretrained_dict)} out of {len(sd['model'])} pretrained layers")
    skipped_layers = set(sd['model'].keys()) - set(pretrained_dict.keys())
    if skipped_layers:
        print(f"Skipped layers due to dimension mismatch: {len(skipped_layers)} layers")

    # 冻结大部分参数，只训练必要的层
    for param in model.patch_embed.parameters():
        param.requires_grad = False

    for blk in model.blocks:
        for param in blk.parameters():
            param.requires_grad = False

    # 训练编码器的最后几层
    num_blocks_to_train = config.num_blocks
    for blk in model.blocks[-num_blocks_to_train:]:
        for param in blk.parameters():
            param.requires_grad = True

    # 训练解码器
    for param in model.decoder_embed.parameters():
        param.requires_grad = True
    
    for param in model.decoder_blocks.parameters():
        param.requires_grad = True

    # 训练新的输出层
    if model.use_dim_reduction:
        for param in model.pre_clip_intermediate1.parameters():
            param.requires_grad = True
        for param in model.pre_clip_intermediate2.parameters():
            param.requires_grad = True
        for param in model.pre_clip_final.parameters():
            param.requires_grad = True
    else:
        for param in model.pre_clip.parameters():
            param.requires_grad = True

    # 计算可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params}")

    model.to(device)

    # 设置分布式采样器
    sampler = torch.utils.data.DistributedSampler(
        train_dataloader.dataset) if torch.cuda.device_count() > 1 else None

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank,
                                        find_unused_parameters=config.use_nature_img_loss)

    # 测试模型前向传播
    print("Testing model forward pass...")
    test_fmri = torch.randn(1, 1, num_voxels).to(device)  # 降低测试batch size
    test_target = torch.randn(1, target_output_dim).to(device)
    model.eval()
    with torch.no_grad():
        try:
            _, _, test_loss, test_pred = model(test_fmri, test_target, mask_ratio=0.75)
            print(f"Test forward pass successful. Loss: {test_loss.item():.4f}, Pred shape: {test_pred.shape}")
        except Exception as e:
            print(f"Test forward pass failed: {e}")
            return
    model.train()

    # 设置训练相关
    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model, log='all', log_freq=1000)

    cor_list = []
    start_time = time.time()
    print('Finetuning MAE on test fMRI ... ...')
    val_loss_min = 10
    patience = 10
    early_stop_counter = 0

    for ep in range(config.num_epoch):
        if torch.cuda.device_count() > 1 and sampler is not None:
            sampler.set_epoch(ep)
            
        # total_loss = train_one_epoch_feat_rdm(model, train_dataloader, optimizer, device, ep, loss_scaler,
        #                                                    logger, config, start_time)
        
        # 要是使用rdm的话
        total_loss = train_one_epoch_feat_rdm(model, train_dataloader, optimizer, device, ep, loss_scaler,
                                                           logger, config, start_time)


        print('Testing MAE on test fMRI ... ...')
        pred, val_loss, cor = validate(model, test_dataloader, optimizer, device, loss_scaler, logger, config, start_time)
        lr = optimizer.param_groups[0]["lr"]
        
        if logger is not None:
            logger.log('train_loss_step', total_loss, step=ep)
            logger.log('val_loss_step', val_loss, step=ep)
            logger.log('lr', lr, step=ep)
            logger.log('cor', np.mean(cor), step=ep)

        cor_list.append(np.mean(cor))

        if val_loss < val_loss_min:
            val_loss_min = val_loss
            np.save(f"{path_root}/decoded/{testsubj}/MAE_{ROI_NAME}_subj{testsubj}_share_scores_{target}_model.npy", pred)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{patience}")
            
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {ep + 1} epochs due to lack of improvement in validation loss.")
            if logger is not None:
                logger.log('earlystop', np.mean(cor), step=ep-10)
            break

    if logger is not None:
        logger.log('max cor', np.max(cor_list), step=config.num_epoch - 1)
        logger.finish()
    return


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
    main(config)

