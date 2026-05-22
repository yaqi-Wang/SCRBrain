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
from config import Config_MBM_finetune_Clip, Config_MBM_fMRI
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from sc_mbm.mae_for_fmri import MAEforFMRI, fMRI2CLIP
from sc_mbm.trainer import train_one_epoch_feat,validate
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."
os.environ['WANDB_MODE'] = "offline"

# ROI_NAME = 'all' # lowvis visual other all

# if ROI_NAME == 'lowvis':
#     DATA_ROI_NAME = 'LVC/epoch55'
# elif ROI_NAME == 'all':
#     DATA_ROI_NAME = 'All_rois/epoch_55'
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
    parser.add_argument("--subject", type=str, default='subj02')
    parser.add_argument('--root_path', type=str)
    # parser.add_argument('--pretrain_mbm_path',
    #                     # default=f'/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_finetune_save/{DATA_ROI_NAME}/checkpoints/checkpoint.pth',
    #                     # default='/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_finetune_save/fmri_data/All_rois/30-05-2025-14-20-14/epoch_90/checkpoints/checkpoint.pth',
    #                     default='/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_finetune_save/fmri_data/All_rois/subj05/epoch_65/checkpoints/checkpoint.pth',
    #                     type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--include_nonavg_test', type=bool)
    parser.add_argument('--ROI_NAME', type=str, default='lowvis')

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

def get_dataloader(fmri_data, feat_data, batch_size=4, shuffle=True, num_workers=4):
    dataset = FMRIAndFeatDataset(fmri_data, feat_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def main(config):
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    ROI_NAME = config.ROI_NAME # lowvis visual other all

    if ROI_NAME == 'lowvis':
        DATA_ROI_NAME = 'LVC'
    elif ROI_NAME == 'all':
        DATA_ROI_NAME = 'All_rois'
    elif ROI_NAME == 'visual':
        DATA_ROI_NAME = 'OnlyVision'
    elif ROI_NAME == 'other':
        DATA_ROI_NAME = 'noVC'
    elif ROI_NAME == '0.25':
        DATA_ROI_NAME = 'All_rois'
    elif ROI_NAME == '0.5':
        DATA_ROI_NAME = 'All_rois'
    elif ROI_NAME == '0.9':
        DATA_ROI_NAME = 'All_rois'


    testsubj=config.subject
    print("test subj: ", testsubj)
    target = 'vdc_clip'
    print("target: ", target)
    path_root = '/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main'

    model_path = '/home/data/wangyaqi/projects/03MAE_latent_diffusion_fMRI/01mind-vis-main/code/results/fmri_finetune_save/fmri_data'
    epoch_num = 'epoch_40'
    # pretrain_mbm_path = f'{model_path}/{DATA_ROI_NAME}/{testsubj}/share/{epoch_num}/checkpoints/checkpoint.pth'
    pretrain_mbm_path = f'{model_path}/All_rois/{testsubj}/share/epoch_last/checkpoints/checkpoint.pth'


    sd = torch.load(pretrain_mbm_path, map_location='cpu')
    config_pretrain = sd['config']

    # output_path = os.path.join(config.root_path, 'results', 'fmri_2_clip',
    #                            '%s_%s' % (config.subject, datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    # # output_path = os.path.join(config.root_path, 'results', 'fmri_finetune')
    # config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
 
    # if config.local_rank == 0:
    #     os.makedirs(output_path, exist_ok=True)
    #     create_readme(config, output_path)

    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config_pretrain.seed)
    np.random.seed(config_pretrain.seed)

    print('pram config:')
    print(f'subject: {testsubj}, target: {target}, ROI: {ROI_NAME}')
    # 加载某一被试的训练/测试 fMRI 信号数据（形状通常是 [样本数, voxel数]）
    fmri_tr = []
    fmri_te = []

    # subjects = [1, 2, 5, 7]
    # train_sets = []
    # for subj in subjects:
    #     train_data = np.load(f'/home/dell/Codes/StableDiffusionReconstruction/mrifeat_new/subj0{subj}/subj0{subj}_all_betas_ave_tr.npy')
    #     train_sets.append(train_data)
    #     fmri_tr = np.concatenate(train_sets, axis=0)
   
    # fmri_tr = np.load(f'{path_root}/mrifeat/{testsubj}/{testsubj}_{ROI_NAME}_rois_betas_ave_tr_aligned.npy')
    # fmri_te = np.load(f'{path_root}/mrifeat/{testsubj}/{testsubj}_{ROI_NAME}_rois_betas_ave_te_aligned.npy')
    fmri_tr = np.load(f'{path_root}/mrifeat/{testsubj}/{testsubj}_all_rois_betas_ave_tr_aligned.npy')
    fmri_te = np.load(f'{path_root}/mrifeat/{testsubj}/{testsubj}_all_rois_betas_ave_te_aligned.npy')    
    # fmri_tr = np.load(f'/home/dell/Codes/StableDiffusionReconstruction/mrifeat_new/{test}/{test}_all_rois_betas_ave_tr.npy')
    # fmri_te = np.load(f'/home/dell/Codes/StableDiffusionReconstruction/mrifeat_new/{test}/{test}_all_rois_betas_ave_te.npy')

    fmri_tr = fmri_tr / 300
    fmri_te = fmri_te / 300

    fmri_tr = pad_to_patch_size(fmri_tr, 16)
    fmri_te = pad_to_patch_size(fmri_te, 16)


    # train_sets=[]
    # for subj in subjects:
    #     train_data = np.load(f'/home/dell/Codes/StableDiffusionReconstruction/nsdfeat/subjfeat/subj0{subj}_ave_c_tr.npy').reshape(-1, 77, 768)
    #     train_sets.append(train_data)
    #     CLIP_tr = np.concatenate(train_sets, axis=0)

    

    # 加载对应的 CLIP 特征
    # 加载对应的 CLIP 特征
    if target == 'vdc_clip':
        CLIP_tr = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_tr.npy').reshape(-1, 77, 768)
        CLIP_te = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_te.npy').reshape(-1, 77, 768)
        train_dataloader = get_dataloader(fmri_tr, CLIP_tr, batch_size=config.batch_size, shuffle=True)
        test_dataloader = get_dataloader(fmri_te, CLIP_te, batch_size=config.batch_size, shuffle=False)

    elif target == 'init_latent':
        init_latent_tr = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_tr.npy')
        init_latent_te = np.load(f'{path_root}/nsdfeat/subjfeat/{testsubj}_ave_{target}_te.npy')
        train_dataloader = get_dataloader(fmri_tr, init_latent_tr, batch_size=16, shuffle=True)
        test_dataloader = get_dataloader(fmri_te, init_latent_te, batch_size=16, shuffle=False)
    # train_dataloader = get_dataloader(fmri_tr, init_latent_tr, batch_size=16, shuffle=True)
    # test_dataloader = get_dataloader(fmri_te, init_latent_te, batch_size=16, shuffle=False)

    # train_dataloader = get_dataloader(fmri_tr, init_latent_tr, batch_size=32, shuffle=True)
    # test_dataloader = get_dataloader(fmri_te, init_latent_te, batch_size=32, shuffle=False)

    # train_set = np.concatenate((train_set,test_set), axis=0)


    # create modelprint(f'train_set shape:{fmri_tr.shape}')
    # 用当前输入的真实长度建模，避免和旧 checkpoint 的 pos_embed 长度绑定在一起
    num_voxels = fmri_tr.shape[1]
    model = fMRI2CLIP(num_voxels=num_voxels, patch_size=config_pretrain.patch_size,
                       embed_dim=config_pretrain.embed_dim, decoder_depth=config.decoder_depth,
                       decoder_embed_dim=config_pretrain.decoder_embed_dim, depth=config_pretrain.depth,
                       num_heads=config_pretrain.num_heads, decoder_num_heads=config_pretrain.decoder_num_heads,
                       mlp_ratio=config_pretrain.mlp_ratio, focus_range=None, use_nature_img_loss=False)
    state_dict = copy.deepcopy(sd['model'])
    state_dict.pop('pos_embed', None)
    state_dict.pop('decoder_pos_embed', None)
    model.load_state_dict(state_dict, strict=False)

    # 只微调 encoder 的最后几层 block，其它部分全部冻结
    for blk in model.blocks:
        blk.requires_grad = False

    for param in model.patch_embed.parameters():
        param.requires_grad = False

    for blk in model.blocks:
        for param in blk.parameters():
            param.requires_grad = False

    num_blocks_to_train = config.num_blocks
    for blk in model.blocks[-num_blocks_to_train:]:
        for param in blk.parameters():
            param.requires_grad = True

    # 初始化可训练参数总数，计算模型的参数量
    trainable_params = 0

    # 遍历模型中的每个参数
    for param in model.parameters():
        # if param.requires_grad:  # 如果参数可训练
        trainable_params += param.numel()  # 累加该参数的元素数量 

    print(f"Total trainable parameters: {trainable_params}")

    # for param in model.decoder_embed.parameters():
    #     param.requires_grad = True
    #
    # for param in model.decoder_blocks.parameters():
    #     param.requires_grad = True
    #
    # for param in model.pre_clip.parameters():
    #     param.requires_grad = True

    model.to(device)
    # model_without_ = model

    sampler = torch.utils.data.DistributedSampler(
        fmri_tr) if torch.cuda.device_count() > 1 else torch.utils.data.RandomSampler(fmri_tr)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank,
                                        find_unused_parameters=config.use_nature_img_loss)

    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, betas=(0.9, 0.95))

    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model, log='all', log_freq=1000)

    cor_list = []
    val_cor_list=[]
    pred_list=[]
    start_time = time.time()
    print('Finetuning MAE on test fMRI ... ...')
    val_loss_min=10
    patience = 10
    early_stop_counter = 0
    print('Training MAE on test fMRI ... ...:'+ f'/MAE_{ROI_NAME}_{testsubj}_share_scores_{target}.npy')

    for ep in range(config.num_epoch):
    # for ep in range(1):
        epoch_save_dir = f'epoch_{ep}'
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(ep)  # to shuffle the data at every epoch
        # total_loss, clip_loss, rdm_loss = train_one_epoch_feat(model, train_dataloader, optimizer, device, ep, loss_scaler, logger, config, start_time)
        total_loss = train_one_epoch_feat(model, train_dataloader, optimizer, device, ep, loss_scaler,
                                                           logger, config, start_time)
        # if (ep % 10 == 0 or ep + 1 == config.num_epoch) and ep != 0 and config.local_rank == 0:

        print('Testing MAE on test fMRI ... ...')
        pred, val_loss,cor = validate(model, test_dataloader, optimizer, device, loss_scaler, logger, config, start_time)
        lr = optimizer.param_groups[0]["lr"]
        logger.log('train_loss_step', total_loss,step=ep)
        # logger.log('train_loss_step', clip_loss, step=ep)
        # logger.log('rdm_loss_step', rdm_loss, step=ep)
        logger.log('val_loss_step', val_loss, step=ep)
        logger.log('lr', lr, step=ep)
        logger.log('cor', np.mean(cor), step=ep)

        cor_list.append(np.mean(cor))

        if val_loss<val_loss_min:
            # np.save(f"{path_root}/decoded/{testsubj}/MAE_{ROI_NAME}_subj02-07_scores_{target}.npy", pred)
            np.save(f"{path_root}/decoded/{testsubj}/MAE_{ROI_NAME}_{testsubj}_share_scores_{target}.npy", pred)
            # save_model(config_pretrain, ep, model, optimizer, loss_scaler, os.path.join(output_path, 'checkpoints'))
            early_stop_counter = 0
        else:
            # 增加早停计数器
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {ep + 1} epochs due to lack of improvement in validation loss.")
            logger.log('earlystop', np.mean(cor), step=ep-10)
            break

    # np.save("/home/dell/Codes/StableDiffusionReconstruction/decoded/subj01/MAE_all_rois_scores_c.npy", pred)
    # np.save("/home/dell/Codes/StableDiffusionReconstruction/decoded/subj01/MAE_init_latent.npy", clip)

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))
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
    config = Config_MBM_finetune_Clip()
    config = update_config(args, config)
    main(config)
