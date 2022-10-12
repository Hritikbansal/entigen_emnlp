# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
import time
# import clip
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
# import more_itertools
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score
from .social_dataset import SocialDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--softmax-temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=256)
    parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--are_classifier_prompts', action = "store_true", default = False, help = "classify prompts")

    parser.add_argument('--image_dump_dir', type = str, default='image dump directory')
    parser.add_argument('--output_fname', type = str, default='mindalle_output_person')

    parser.add_argument('--intervention', type = str, default=None)
    parser.add_argument('--entity', type = str, default=None)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device_id', type=int, default=3)

    args = parser.parse_args()
    print(args)

    # Setup
    assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

    set_seed(args.seed)

    model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.

    device = f'cuda:{args.device_id}'

    # Load checkpoint
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        model.to(device=device)
        print('Model loaded from {}'.format(args.ckpt_path))
    else:
        model.to(device=device)
        print('Model loaded from pretrained weights.')


    dataset = SocialDataset(
        tokenizer=model.tokenizer,
        intervention = args.intervention,
        are_classifier_prompts = args.are_classifier_prompts,
        entity = args.entity
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.text_collate_fn,
        num_workers=4,
    )

    image_dump_dir = f'{args.image_dump_dir}/{args.output_fname}'
    os.makedirs(image_dump_dir, exist_ok=True)
    print('Image dump at: ', image_dump_dir)

    output_dir = image_dump_dir
    start = time.time()

    for j in range(9):
        for batch in tqdm(loader):
            text_tokens = batch['input_ids']
            text_tokens = text_tokens.to(device)

            B = len(text_tokens)

            images = model.sampling(prompt="",
                                    tokens=text_tokens,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    softmax_temperature=args.softmax_temperature,
                                    num_candidates=B,
                                    device=device,
                                    is_tqdm=False,
                                    ).cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            print(images.shape)

            for i in range(len(images)):
                im = Image.fromarray((images[i]*255).astype(np.uint8))

                caption = batch['caption'][i]

                caption_dir = os.path.join(image_dump_dir, caption)
                os.makedirs(caption_dir, exist_ok = True)

                fname = f"{j}.jpg"
                out_fname = os.path.join(caption_dir, fname)
                im.save(out_fname)
                
    end = time.time()
    print(f'Total Time Taken: {end-start} seconds')