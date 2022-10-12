from pathlib import Path
from random import randint, choice
from re import I

import PIL
from PIL.Image import ImageTransformHandler

import json
from copy import deepcopy
import numpy as np
import torch

import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms



class SocialDataset(Dataset):
    def __init__(self,
                 tokenizer=None,
                 intervention = None,
                 are_classifier_prompts = None,
                 entity = 'profession'
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        social_data_dir = Path(__file__).resolve().parent.joinpath('social_bias')

        prompts = []

        for prompt_name in [entity]:
            with open(social_data_dir.joinpath(f"neutral_prompts_{prompt_name}.json"), 'r') as f:
                prompt_data = json.load(f)
                if are_classifier_prompts:
                    prompts += prompt_data['classifier_prompts']
                else:
                    prompts += prompt_data['neutral_prompts']

        self.data = prompts
        self.entity = entity
        self.tokenizer = tokenizer
        self.intervention = intervention

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        caption = self.data[ind]
        tbr = caption.split(' ')[-1] if self.entity == 'profession' else caption.split(' ')[-2] if self.entity == 'color' else  caption.split(' ')[-2] if self.entity == 'others' else ''
        caption_aug = caption + " " + self.intervention.replace('[mask]', tbr) if self.intervention else caption
        tokens = self.tokenizer.encode(caption_aug)
        input_ids = torch.LongTensor(tokens.ids)

        # out = {
        #     'caption': caption + ' person',
        #     'input_ids': input_ids
        # }
        out = {
            'caption': caption,
            'input_ids': input_ids
        }

        return out

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b['input_ids']) for b in batch])

        batch_datum = {
            'caption': [],
            'input_ids': torch.LongTensor(B, L),
        }

        for i, datum in enumerate(batch):
            batch_datum['caption'].append(datum['caption'])
            batch_datum['input_ids'][i] = datum['input_ids']

        return batch_datum
