import argparse
import os
import h5py
import scipy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pytorch_lightning import seed_everything
import torchvision.transforms as tvtrans
import time

from nsd_access import NSDAccess
from versatile_diffusion.lib.cfg_helper import model_cfg_bank
from versatile_diffusion.lib.model_zoo import get_model
from versatile_diffusion.lib.model_zoo.ddim_vd import DDIMSampler_VD


def load_vd_model(vd_model_path, device_str, cfgm_name='vd_noema', use_fp16=True):
    print(f"Loading Versatile Diffusion model from {vd_model_path} on {device_str}")
    device = torch.device(device_str)
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(vd_model_path, map_location="cpu")
    if 'module.' in list(sd.keys())[0] and not hasattr(net, 'module'):
        sd = {k[7:]: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.clip.to(device)
    net.autokl.to(device)
    net.model.diffusion_model.to(device)
    if use_fp16:
        net.autokl.half()
        net.model.diffusion_model.half()
    net.eval()
    print("Versatile Diffusion model loaded.")
    return net

def regularize_image_vd(x, target_size=512):
    BICUBIC = Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).convert("RGB").resize([target_size, target_size], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, Image.Image):
        x = x.convert("RGB").resize([target_size, target_size], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = Image.fromarray(x).convert("RGB").resize([target_size, target_size], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
    else:
        raise ValueError(f"Unknown image type: {type(x)}")
    if x.ndim == 3 and (x.shape[1] != target_size or x.shape[2] != target_size):
        x = tvtrans.ToPILImage()(x)
        x = x.convert("RGB").resize([target_size, target_size], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif x.ndim != 3:
        raise ValueError("Tensor input must be 3D (CHW) for resizing.")
    return x

def main(paras_fmri, subject):
    parser = argparse.ArgumentParser()
    # ...existing parser arguments...
    parser.add_argument("--imgidx", default=[808, 982], nargs="*", type=int, help="start and end imgs (e.g., 0 100). Process at least one image to see stats.") # 修改默认值为处理少量图像，方便查看统计信息
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--vd_model_path", type=str, default='/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/codes/utils/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth', help="Path to the Versatile Diffusion .pth model file")
    parser.add_argument("--vd_cfgm_name", type=str, default='vd_noema', help="VD config name")
    parser.add_argument("--use_fp16", action='store_true', default=True, help="Use fp16 inference")
    parser.add_argument("--no_fp16", dest="use_fp16", action='store_false', help="Disable fp16 inference")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--strength", type=float, default=0.5, help="Guidance strength")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of generations per image")
    parser.add_argument("--blend_ratio", type=float, default=0.5, help="Vision-text blend ratio") # 0.75
    parser.add_argument("--roi_type", type=str, default='all', help="Region of interest type") # 0.75
    # parser.add_argument(
    #     "--subject",
    #     default='subj01',
    #     type=str,
    #     help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    # )
    args = parser.parse_args()

    seed_everything(args.seed)
    device_str = f"cuda:{args.gpu}"
    device = torch.device(device_str)
    n_iter = args.n_iter
    root_path = '/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main'

    net = load_vd_model(args.vd_model_path, device_str, cfgm_name=args.vd_cfgm_name, use_fp16=args.use_fp16)
    sampler = DDIMSampler_VD(net)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=True)

    # NSD 原始数据所在路径
    nsd_data_path = '/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data'
    nsd_expdesign = scipy.io.loadmat(f'{nsd_data_path}/nsddata/experiments/nsd/nsd_expdesign.mat')
    # Note that mos of them are 1-base index!
    # This is why I subtract 1
    sharedix = nsd_expdesign['sharedix'] -1 
    stims_ave = np.load(f'{root_path}/mrifeat_0526/{subject}/{subject}_stims_ave.npy')
    
    # 这里用于选择测试集还是训练集
    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1
    
    
    # 进行图像重建，确定一个范围
    test_indices = np.where(tr_idx==0)[0]
    
    
    t_enc = int(args.strength * args.ddim_steps)
    print(f"Target t_enc is {t_enc} steps")

    # 输入潜特征空间
    file_root_path= f'{root_path}/decoded'
    # perturb_out_path = '/home/data/wangyaqi/projects/12AlighnmentCognitive/perturb_out'
    # paras_fmri = 's0_vis' # s0_vis s1_vis s2a_vis s2b_vis s3_vis s4_vis
    # paras_fmri = paras_fmri
    subject = subject

    # image_source_dir = f"{file_root_path}/vdvae_{subject}_{paras_fmri}_{paras_theta}_low"
    # image_source_dir = f"{file_root_path}/image-cvpr/{subject}/{subject}_{paras_fmri}_share_sdvae_low_fmri_latent"
    image_source_dir = f"{file_root_path}/image-cvpr/{subject}/{subject}_all_sdvae_low_fmri_latent"
   
    #保存重建图像和原始图像
    # output_dir = f"{file_root_path}/vdvae_{subject}_{paras_fmri}_{paras_theta}_recon_vd"
    output_dir = f"{file_root_path}/image-cvpr/{subject}/{subject}_{paras_fmri}_versatile_recon_fmri_share_Ridge"
    os.makedirs(output_dir, exist_ok=True)   

    scores_c_path = f'{root_path}/decoded/{subject}/MAE_all_{subject}_share_scores_vdc_clip.npy'
    scores_c_all = np.load(scores_c_path)
    scores_vision_path = f'{root_path}/decoded/{subject}/Ridge_all_{subject}_share_scores_vdv_latent.npy'
    scores_vision_all = np.hstack(np.split(np.load(scores_vision_path), 3, axis=0))

    # 为了获取稳定的统计数据，可以考虑对多张图片取平均，但这里我们先看单张图片的统计
    # 或者只处理一张图片来查看其潜特征统计
    # 修改 imgidx 的默认值，例如 default=[85, 86] 来只处理一张图片进行检查

    for current_selection_idx in tqdm(range(args.imgidx[0], args.imgidx[1])):

        print(f"\nProcessing image {current_selection_idx}...")
        print('-'*50)
        
         # 获取单个测试图像的索引
        single_imgidx_te = test_indices[current_selection_idx]
        s_idx_in_nsd = stims_ave[single_imgidx_te] # 这现在是一个标量
       

        # s_idx_in_nsd 是图像在NSD 73k中的索引
        # current_selection_idx是图像在0-1000中的索引
        print(f"\n--- Processing image NSD index: {current_selection_idx}-{s_idx_in_nsd:06} ---")
        # img_data_np1 = nsda.read_images(s_idx_in_nsd) # img_data_np 是原始图像的numpy数组

        # image_filename = f"{current_selection_idx:04d}.png" # Assuming 5-digit padding like 00000.png
        image_filename = f"{current_selection_idx:05d}_latent.png" # Assuming 5-digit padding like 00000.png
        image_path = os.path.join(image_source_dir, image_filename)
        img_pil = Image.open(image_path).convert('RGB')
        img_data_np = np.array(img_pil)

        # 确认一下图像是不是能够对的上
        # 将img_data_np1和img_data_np分别保存成图片
        # img_data_np1_pil = Image.fromarray(img_data_np1)
        # img_data_np1_pil.save(os.path.join(output_dir, f"{current_selection_idx:05d}_org.png"))

        # 先不保存原图

        # img_data_np_pil = Image.fromarray(img_data_np)
        # img_data_np_pil.save(os.path.join(output_dir, f"{current_selection_idx:05d}_vd.png"))


        
        # 1. 使用 regularize_image_vd 将其处理成 VD 期望的输入尺寸
        # regularize_image_vd 内部处理 PIL Image 或 ndarray 到 Tensor (0 to 1 range)
        # 然后 unsqueeze(0) 增加 batch 维度, to(device), 归一化到 [-1, 1]
        image_tensor = regularize_image_vd(img_data_np).unsqueeze(0).to(device) * 2.0 - 1.0
        
        if args.use_fp16:
            image_tensor = image_tensor.half()

        # 确认图像张量的形状是否正确 (VD 通常期望 512x512)
        print(f"Input image_tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
        assert image_tensor.shape == (1, 3, 512, 512), f"Image tensor shape invalid: {image_tensor.shape}"
        
        # 2. 将其输入到 vd_net.autokl_encode() 中，得到 VD VAE 编码后的潜特征
        with torch.no_grad(): # 确保在推理模式下不计算梯度
            init_latent = net.autokl_encode(image_tensor) # init_latent 是 VD 原生的潜特征
        if args.use_fp16:
            init_latent = init_latent.half()
        
        # 3. 检查这个原生潜特征的 shape、mean()、std()、min()、max()
        print(f"VD Native init_latent shape: {init_latent.shape}")
        print(f"VD Native init_latent dtype: {init_latent.dtype}")
        print(f"VD Native init_latent mean: {init_latent.mean().item():.4f}")
        print(f"VD Native init_latent std: {init_latent.std().item():.4f}")
        print(f"VD Native init_latent min: {init_latent.min().item():.4f}")
        print(f"VD Native init_latent max: {init_latent.max().item():.4f}")
        
        # 确认潜特征的空间维度 (VD 通常是 64x64)
        assert init_latent.shape[-2:] == (64, 64), f"Latent spatial dimensions invalid: {init_latent.shape}, expected H, W to be 64, 64"
        assert init_latent.shape[1] == 4, f"Latent channel dimension invalid: {init_latent.shape}, expected C to be 4" # VD VAE 通常是4通道

        # --- 后续的重建过程 (您的脚本中已有的部分) ---
        # 这部分代码使用 VD 原生的 init_latent 进行重建，
        # 如果这里能生成清晰的图片，说明 VD 模型和采样流程本身是正常的。
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc], device=device))
        if args.use_fp16:
            z_enc = z_enc.half()

        # 使用无条件（空）的文本和视觉引导作为基线测试
        # 注意：VD的CLIP输入通常是224x224
        dummy_clip_img_input = torch.zeros((1, 3, 224, 224), device=device) 
        if args.use_fp16:
            dummy_clip_img_input = dummy_clip_img_input.half()

        with torch.no_grad():
            uim = net.clip_encode_vision(dummy_clip_img_input) # 无条件的视觉嵌入
            utx = net.clip_encode_text("") # 无条件的文本嵌入

        # 有条件的文本嵌入

        # scores_c_path = f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/extracted_features/{subject}/{subject}_ave_vdc_clip_te.npy'
        # scores_c_path = f'{root_path}/decoded/congnitive/c_vd_clip/{subject}/{subject}_vdc_clip_{paras_fmri}_wood.npy'
        scores_c = scores_c_all[current_selection_idx,:].reshape(77,768)
        c = torch.as_tensor(scores_c, dtype=torch.float32, device=device).unsqueeze(0)
        

        # 有条件的图像引导嵌入
        # scores_vision_path = f'{perturb_out_path}/{subject}_vdv_latent_{paras_fmri}/{paras_theta}.npy' 
        # scores_vision = np.load(scores_vision_path)[current_selection_idx,:].reshape(257,768)
        scores_vision = scores_vision_all[current_selection_idx,:].reshape(257,768)
        vim = torch.as_tensor(scores_vision, dtype=torch.float32, device=device).unsqueeze(0)


        if args.use_fp16:
            uim = uim.half()
            utx = utx.half()
            c = c.half()
            vim = vim.half()


        print(f"Shape of uim (unconditional vision): {uim.shape}, dtype: {uim.dtype}")
        print(f"Shape of utx (unconditional text): {utx.shape}, dtype: {utx.dtype}")
        print(f"Shape of z_enc (stochastic encoded latent): {z_enc.shape}, dtype: {z_enc.dtype}")
        print(f"Shape of c (conditional text): {c.shape}, dtype: {c.dtype}")
        print(f"Shape of vim (conditional vision): {vim.shape}, dtype: {vim.dtype}")

        # 使用 sampler.decode_dc 进行解码
        # 注意: first_conditioning 和 second_conditioning 需要是列表，即使只有一个条件
        # 对于无条件生成或仅用 latent 重建，通常会将 uim 和 utx 作为条件传入
        # 这里的混合比例 (1 - args.blend_ratio) 可能需要根据您的具体意图调整
        # 如果只想测试 latent 到 image 的解码，可以简化条件
        for sample_idx in range(n_iter):
            with torch.no_grad():
                z = sampler.decode_dc(
                    x_latent=z_enc, # 噪声化的潜变量
                    first_conditioning=[uim, vim],    # [unconditional_visual, conditional_visual]
                    second_conditioning=[utx, utx],   # [unconditional_text, conditional_text]
                    # first_conditioning=[uim, uim],    # [unconditional_visual, conditional_visual]
                    # second_conditioning=[utx, utx],   # [unconditional_text, conditional_text]
                    t_start=t_enc,
                    unconditional_guidance_scale=args.scale,
                    xtype='image', # 表明 x_latent 是图像潜变量
                    first_ctype='vision', # 表明 first_conditioning 是视觉条件
                    second_ctype='prompt', # 表明 second_conditioning 是文本条件
                    mixed_ratio=(1 - args.blend_ratio), # 控制视觉和文本条件的混合
                )

            # 将解码后的潜变量 z 通过 VAE 解码器转换回图像空间
            with torch.no_grad():
                x_reconstructed_pixels = net.autokl_decode(z)
            
            # 后处理：反归一化到 [0,1] 并转换为 PIL Image
            x_reconstructed_pixels = torch.clamp((x_reconstructed_pixels + 1.0) / 2.0, min=0.0, max=1.0)
            

            
            # for i in range(x_reconstructed_pixels.shape[0]): # 循环处理batch中的每张图 (虽然这里batch_size=1)
            #     recon_pil_img = tvtrans.ToPILImage()(x_reconstructed_pixels[i].cpu())
            #     recon_pil_img.save(os.path.join(output_dir, f"{current_selection_idx:05d}_{sample_idx:003}.png"))
            
            for i in range(x_reconstructed_pixels.shape[0]): # 循环处理batch中的每张图 (虽然这里batch_size=1)
                recon_pil_img = tvtrans.ToPILImage()(x_reconstructed_pixels[i].cpu())
                recon_pil_img.save(os.path.join(output_dir, f"{current_selection_idx:04d}.png"))
            
            # original_pil_img = Image.fromarray(img_data_np) # 从原始numpy数组创建PIL图像
            # original_pil_img.save(os.path.join(output_dir, f"{current_selection_idx:06d}_origin.png"))
            
            # print(f"Saved original and reconstructed images for NSD index {s_idx_in_nsd:05d} to {output_dir}")

            # 如果只想检查一张图片的统计数据，可以在这里 break
            # break


if __name__ == '__main__':
    subject_list = ['subj07'] # 
    paras_fmri_list = ['vision'] # 'vdvae' 'orgreg' 'mlpmse' 'cnnmse'
    
    for subject in subject_list:
        for paras_fmri in paras_fmri_list:
            stat_time = time.time()
            main(paras_fmri, subject)
            print(f"Total processing time: {time.time() - stat_time:.2f} seconds")
