import os
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
from utils import ptp_utils,seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from Freeprompt.freeprompt import SelfAttentionControlEdit,AttentionStore,EmptyControl
from Freeprompt.freeprompt_utils import register_attention_control_new
import json

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:4') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler).to(device)
tokenizer = ldm_stable.tokenizer


# +
@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    register_attention_control_new(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)

    return images, x_t
# -



g_cpu = torch.Generator().manual_seed(42)

data_dir = "./datasets"
data_name = "Car_fake_edit"

# +
out_dir = "./outputs"
source_output_dir=f"{out_dir}/{data_name}_p2p/source/"
target_output_dir=f"{out_dir}/{data_name}_p2p/target/"

os.makedirs(source_output_dir, exist_ok = True)
os.makedirs(target_output_dir, exist_ok = True)
# -

with open(f"{data_dir}/{data_name}.json", "r") as f:
    data = json.load(f)

idx = 0
for item in tqdm(data):
    source_prompt = item["soure_prompt"]   
    target_prompt = item["edit_prompt"]
    
    start_code = torch.randn([1, 4, 64, 64], device=device)
    start_code = start_code.expand(2, -1, -1, -1)

    prompts = [source_prompt]
    controller = AttentionStore()
    image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)

    prompts = [source_prompt, target_prompt]
    
    self_replace_steps = .8
    NUM_DIFFUSION_STEPS = 50
    controller = SelfAttentionControlEdit(prompts, NUM_DDIM_STEPS, self_replace_steps=self_replace_steps)
    image, _ = run_and_display(prompts, controller, latent=x_t)
    results = ptp_utils.view_images([image[0],image[1]])
    
    
    
    results[0].save(os.path.join(source_output_dir, f'{str(idx)}_{str(source_prompt)}.jpg'))
    results[1].save(os.path.join(target_output_dir, f'{str(idx)}_{str(target_prompt)}.jpg'))
    
    idx+=1


