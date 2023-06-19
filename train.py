import torch
from tools import vae_tools
from diffusers import AutoencoderKL
vae_path = "vae-ft-mse-840000-ema-pruned.ckpt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16

def main():
    # vae = AutoencoderKL.from_pretrained("D:\\Dev\\stable-diffusion-webui\\models\\VAE\\vae-ft-mse-840000-ema-pruned.ckpt")
    vae = vae_tools.load_vae(vae_path, dtype=dtype)
    # set decoder to train
    vae.decoder.requires_grad = True
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)

    #emable the decoder to train
    vae.decoder.requires_grad = True


    print(vae)


if __name__ == "__main__":
    main()
