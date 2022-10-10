from pathlib import Path
from random import randint, choice
from re import I
import json
from copy import deepcopy
import numpy as np

class SocialDataset():
    def __init__(self,
                 processor=None,
                 intervention = None,
                 are_classifier_prompts = None,
                 entity = 'profession',
                 batch_size = 8
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        social_data_dir = Path(__file__).resolve().parent.joinpath('social_bias')

        prompts = []

        # for prompt_name in ['objects', 'politics', 'profession', 'others']:
        # for prompt_name in ['profession']:
        for prompt_name in [entity]:
            with open(social_data_dir.joinpath(f"neutral_prompts_{prompt_name}.json"), 'r') as f:
                prompt_data = json.load(f)
                if are_classifier_prompts:
                    prompts += prompt_data['classifier_prompts']
                else:
                    prompts += prompt_data['neutral_prompts']

        self.data = prompts
        self.entity = entity
        self.processor = processor
        self.intervention = intervention
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        
        prompts = self.data[ind * self.batch_size: (ind + 1) * self.batch_size]
        tbr = prompts[0].split(' ')[-1] if self.entity == 'profession' else prompts[0].split(' ')[-2] if self.entity == 'color' else  prompts[0].split(' ')[-2] if self.entity == 'others' else ''
        augmented_prompts = list(map(lambda prompt: prompt + " " + self.intervention.replace('[mask]', tbr), prompts)) if self.intervention else prompts
        tokenized = self.processor(augmented_prompts)

        out = {
            'tokenized': tokenized,
            'captions': prompts
        }

        return out



