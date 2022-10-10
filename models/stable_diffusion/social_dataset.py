from pathlib import Path
from random import randint, choice
from re import I
import json
from copy import deepcopy
import numpy as np

class SocialDataset():
    def __init__(self,
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
        self.intervention = intervention

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        caption = self.data[ind]
        replacement = caption.split(' ')[-1] if self.entity == 'profession' else caption.split(' ')[-2] if self.entity == 'color' else  caption.split(' ')[-2] if self.entity == 'others' else ''
        caption_aug = caption + " " + self.intervention.replace('[mask]', replacement) if self.intervention else caption
        out = {
            'caption': caption,
            'caption_aug': caption_aug,
        }

        return out

    def text_collate_fn(self, batch):
        batch_datum = {
            'caption': [],
            'caption_aug': [],
        }

        for i, datum in enumerate(batch):
            batch_datum['caption'].append(datum['caption'])
            batch_datum['caption_aug'].append(datum['caption_aug'])

        return batch_datum
