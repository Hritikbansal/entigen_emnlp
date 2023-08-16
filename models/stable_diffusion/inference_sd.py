import os
import sys
import torch
import random
import argparse
from torch.cuda.amp import autocast
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from social_dataset import SocialDataset

parser = argparse.ArgumentParser()

parser.add_argument('--intervention', type = str, default=None)
parser.add_argument('--are_classifier_prompts', action = 'store_true')
parser.add_argument('--entity', type = str, default=None)
parser.add_argument('--image_dump_dir', type = str, default='image dump directory')
parser.add_argument('--output_fname', type = str, default='mindalle_output_person')
parser.add_argument('--device_id', type=int, default=0)

args = parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token = True).to(args.device_id)
print('model loaded!')

dataset = SocialDataset(
            intervention = args.intervention,
            are_classifier_prompts = args.are_classifier_prompts,
            entity = args.entity
        )

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    drop_last=False,
    collate_fn=dataset.text_collate_fn,
)

image_dump_dir = f'{args.image_dump_dir}/{args.output_fname}'
os.makedirs(image_dump_dir, exist_ok=True)

n_seeds = 9

for seed in tqdm(range(n_seeds)):
    generator = torch.Generator(f"cuda:{args.device_id}").manual_seed(seed)
    with autocast():
        for batch in tqdm(loader):
            # print("\n\n keys are:", pipe(batch['caption_aug'], generator = generator).keys(), "\n\n")
            # /n/n keys are: odict_keys(['images', 'nsfw_content_detected']) /n/n 
            # images = pipe(batch['caption_aug'], generator = generator)["sample"]
            images = pipe(batch['caption_aug'], generator = generator)["images"]
            for i in range(len(images)):
                caption = batch['caption'][i]
                caption_dir = os.path.join(image_dump_dir, caption)
                os.makedirs(caption_dir, exist_ok = True)

                fname = f"{seed}.jpg"
                out_fname = os.path.join(caption_dir, fname)
                images[i].save(out_fname)