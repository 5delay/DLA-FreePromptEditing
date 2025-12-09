# -*- coding: utf-8 -*-
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from diffusers import DDIMScheduler
from tqdm import tqdm
from Freeprompt.diffuser_utils import FreePromptPipeline
from Freeprompt.freeprompt_utils import register_attention_control_new
from torchvision.utils import save_image
from torchvision.io import read_image
from Freeprompt.freeprompt import SelfAttentionControlEdit,AttentionStore




device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = FreePromptPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)


# +
# def load_image(image_path, device):
#     image = read_image(image_path)
#     image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
#     image = F.interpolate(image, (512, 512))
#     image = image.to(device)
#     return image
# -

def load_image(image_path, device):
    image = read_image(image_path).float()  # [C, H, W]

    # --- FIX: grayscale → RGB 변환 ---
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # [1,H,W] → [3,H,W]
    elif image.shape[0] > 3:
        image = image[:3]  # RGBA 등은 RGB만 사용
    # ----------------------------------

    image = image.unsqueeze(0) / 127.5 - 1.0  # [1,3,H,W], range [-1,1]
    image = F.interpolate(image, (512, 512))
    return image.to(device)



self_replace_steps = .8
NUM_DIFFUSION_STEPS = 50

data_dir = "./datasets"
data_name = "ImageNet_real_edit"

# +
out_dir = "./outputs"
source_output_dir=f"{out_dir}/{data_name}/source/"
target_output_dir=f"{out_dir}/{data_name}/target/"

os.makedirs(source_output_dir, exist_ok = True)
os.makedirs(target_output_dir, exist_ok = True)
# -

with open(f"{data_dir}/{data_name}.json", "r") as f:
    data = json.load(f)

idx = 0
for item in tqdm(data):
    image_file = item["image_name"].split('/')[-1]
    image_path = f"../data/ImageNet/{image_file}"
    
    source_prompt = item["soure_prompt"]
    target_prompt = item["edit_prompt"]
    
    if idx <175: 
        idx+=1
        continue
    if not os.path.exists(image_path): 
        print("Missing:", image_path)
        continue
        
    source_image = load_image(image_path, device).to(device)
    
    source_prompt = ""
    
    start_code, latents_list = pipe.invert(source_image,
                                        source_prompt,
                                        guidance_scale=7.5,
                                        device=device,
                                        num_inference_steps=50,
                                        return_intermediates=True)
    latents = torch.randn(start_code.shape, device=device)

    prompts = [source_prompt, target_prompt]

    start_code = start_code.expand(len(prompts), -1, -1, -1)
    controller = SelfAttentionControlEdit(prompts, NUM_DIFFUSION_STEPS, self_replace_steps=self_replace_steps)

    register_attention_control_new(pipe, controller)

    results = pipe(prompts,latents=start_code,guidance_scale=7.5, device=device,
                        ref_intermediate_latents=latents_list)
    
    save_image(results[0], os.path.join(source_output_dir, f'{str(idx)}_{str(source_prompt)}.jpg'))
    save_image(results[1], os.path.join(target_output_dir, f'{str(idx)}_{str(target_prompt)}.jpg'))
   
    idx+=1
