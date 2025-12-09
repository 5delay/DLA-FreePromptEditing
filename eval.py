import os 
import torch
import random
import json
import glob
from tqdm import tqdm
import numpy as np

from torch.nn.functional import cosine_similarity
import torch
from PIL import Image
import clip

device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

model, preprocess = clip.load("ViT-B/32", device=device)

data_dir = "./datasets"
data_name = "ImageNet_real_edit"

# +
out_dir = "./outputs"
folder_name =  'ImageNet_real_edit_nullText' #'Car_fake_edit_p2p'
source_output_dir=f"{out_dir}/{folder_name}/source/"

if folder_name == 'ImageNet_real_edit_nullText': source_output_dir=f"{out_dir}/{data_name}/source/"
target_output_dir=f"{out_dir}/{folder_name}/target/"
# -

with open(f"{data_dir}/{data_name}.json", "r") as f:
    data = json.load(f)

cs = []
cds = []

# +
idx = 0
for item in tqdm(data):
    source_prompt = item["soure_prompt"]   
    target_prompt = item["edit_prompt"]
    
    try:
        src_img_path = glob.glob(os.path.join(source_output_dir, f"{idx}_*"))[0]
        tgt_img_path = glob.glob(os.path.join(target_output_dir, f"{idx}_*"))[0]

        src_image = Image.open(src_img_path)
        tgt_image = Image.open(tgt_img_path)
        
    except Exception as e: 
        idx +=1
        continue
        
    # --- Encoding ---
    with torch.no_grad():
        enc_src_image = model.encode_image(preprocess(src_image).unsqueeze(0).to(device))
        enc_tgt_image = model.encode_image(preprocess(tgt_image).unsqueeze(0).to(device))
        
        enc_src_prompt = model.encode_text(clip.tokenize([source_prompt]).to(device))
        enc_tgt_prompt = model.encode_text(clip.tokenize([target_prompt]).to(device))
        
    enc_src_image /= enc_src_image.norm(dim=-1, keepdim=True)
    enc_tgt_image /= enc_tgt_image.norm(dim=-1, keepdim=True)
    
    enc_src_prompt /= enc_src_prompt.norm(dim=-1, keepdim=True)
    enc_tgt_prompt /= enc_tgt_prompt.norm(dim=-1, keepdim=True)
    
    cs_similarity = (enc_tgt_image @ enc_tgt_prompt.T).squeeze().item()
    
    
    delta_image = enc_tgt_image - enc_src_image
    delta_prompt = enc_tgt_prompt - enc_src_prompt
    
    delta_image /= delta_image.norm(dim=-1, keepdim=True) + 1e-6
    delta_prompt /= delta_prompt.norm(dim=-1, keepdim=True) + 1e-6
    
    cds_similarity = (delta_image @ delta_prompt.T).squeeze().item()
    
    cs.append(cs_similarity)
    cds.append(cds_similarity)
    idx+=1

cs_avg = sum(cs) / len(cs)
cds_avg = sum(cds) / len(cds)
print(data_name, cs_avg, cds_avg)
# -


