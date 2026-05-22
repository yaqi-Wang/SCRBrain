import os
import numpy as np
import torch
import torchvision.transforms as tvtrans
from PIL import Image
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from torch.utils.data import Dataset
from tqdm import trange
import argparse

def regularize_image(x):
    """Resize and normalize image to 512x512."""
    BICUBIC = Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        raise ValueError("Unknown image type")
    assert x.shape[1] == 512 and x.shape[2] == 512, "Wrong image size"
    return x

def load_vd_model():
    """Load Versatile Diffusion model and sampler."""
    cfgm_name = 'vd_noema'
    sampler_class = DDIMSampler_VD
    pth = '/home/furkan/Versatile-Diffusion/pretrained/vd-four-flow-v1-0-fp16.pth'
    
    # Load model configuration and weights
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)
    
    # Initialize sampler
    sampler = sampler_class(net)
    
    # Move components to GPU
    net.clip.cuda(0)
    net.autokl.cuda(0).half()
    sampler.model.model.diffusion_model.device = 'cuda:1'
    sampler.model.model.diffusion_model.half().cuda(1)
    
    return net, sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgidx", required=True, type=int, help="Image index")
    parser.add_argument("--gpu", required=True, type=int, help="GPU ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subject", required=True, type=str, help="Subject name")
    parser.add_argument("--method", required=True, type=str, help="cvpr or text or gan")
    opt = parser.parse_args()

    # Set random seed
    torch.manual_seed(opt.seed)

    # Load Versatile Diffusion model and sampler
    net, sampler = load_vd_model()

    # Load predicted latent features
    roi_latent = 'early'
    latent_path = f'../../decoded/{opt.subject}/{opt.subject}_{roi_latent}_scores_init_latent.npy'
    scores_latent = np.load(latent_path)
    imgarr = torch.Tensor(scores_latent[opt.imgidx, :].reshape(4, 40, 40)).unsqueeze(0).to('cuda:1')

    # Set sampling parameters
    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0.0
    scale = 7.5
    xtype = 'image'
    ctype = 'prompt'
    h, w = 512, 512
    shape = [n_samples, 4, h // 8, w // 8]

    # Prepare unconditional conditioning
    u = None
    if scale != 1.0:
        dummy = ''
        u = net.clip_encode_text(dummy)
        u = u.cuda(1).half()

    # Perform sampling
    print("Generating image...")
    z, _ = sampler.sample(
        steps=ddim_steps,
        shape=shape,
        conditioning=imgarr,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=u,
        xtype=xtype,
        ctype=ctype,
        eta=ddim_eta,
        verbose=False,
    )

    # Decode latent to image
    z = z.cuda(0)
    x = net.autokl_decode(z)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    x = [tvtrans.ToPILImage()(xi) for xi in x]

    # Save generated image
    outdir = f'../../decoded/image-{opt.method}/{opt.subject}/'
    os.makedirs(outdir, exist_ok=True)
    sample_path = os.path.join(outdir, f"samples")
    os.makedirs(sample_path, exist_ok=True)
    output_file = os.path.join(sample_path, f"{opt.imgidx:05}.png")
    x[0].save(output_file)
    print(f"Image saved to {output_file}")

if __name__ == "__main__":
    main()