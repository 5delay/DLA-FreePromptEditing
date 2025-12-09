# -*- coding: utf-8 -*-
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import random
import json
from tqdm import tqdm
from diffusers import DDIMScheduler
import numpy as np
from Freeprompt.diffuser_utils import FreePromptPipeline
from Freeprompt.freeprompt_utils import register_attention_control_new
from torchvision.utils import save_image
from torchvision.io import read_image
from Freeprompt.freeprompt import SelfAttentionControlEdit,AttentionStore

# +
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
model_path = "runwayml/stable-diffusion-v1-5"

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = FreePromptPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)


# -

def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    if seed is None:
        seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


# +
seed = 42
seed_everything(seed)

self_replace_steps = .8
NUM_DIFFUSION_STEPS = 50
# -

data_dir = "./datasets"
data_name = "Car_fake_edit"

# +
out_dir = "./outputs"
source_output_dir=f"{out_dir}/{data_name}_/source/"
target_output_dir=f"{out_dir}/{data_name}_/target/"

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

    latents = torch.randn(start_code.shape, device=device)
    prompts = [source_prompt, target_prompt]

    start_code = start_code.expand(len(prompts), -1, -1, -1)
    controller = SelfAttentionControlEdit(prompts, NUM_DIFFUSION_STEPS, self_replace_steps=self_replace_steps)

    register_attention_control_new(pipe, controller)

    results = pipe(prompts,latents=start_code, guidance_scale=7.5, device=device)

    
    source_prompt.replace(" ", "_")
    target_prompt.replace(" ", "_")
    
    save_image(results[0], os.path.join(source_output_dir, f'{str(idx)}_{str(source_prompt)}.jpg'))
    save_image(results[1], os.path.join(target_output_dir, f'{str(idx)}_{str(target_prompt)}.jpg'))
    
    idx+=1



