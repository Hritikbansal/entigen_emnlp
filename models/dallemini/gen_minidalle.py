import jax
import jax.numpy as jnp
import os
import wandb
import argparse
from social_dataset import SocialDataset

os.environ["WANDB_API_KEY"] = "Add your API key"

parser = argparse.ArgumentParser()

parser.add_argument('--intervention', type = str, default=None)
parser.add_argument('--entity', type = str, default=None)
parser.add_argument('--image_dump_dir', type = str, default='image dump directory')
parser.add_argument('--output_fname', type = str, default='minidalle_output_person')
parser.add_argument('--device_id', type=str, default="2")
parser.add_argument('--batch_size', type=int, default=8)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
os.environ['TOKENIZERS_PARALLELISM'] = "False"

wandb.init(project = "minidalle", config = vars(args), entity = "sat-transformers")
wandb.run.name = args.entity + " " + args.output_fname
wandb.run.name = wandb.run.name + " " + args.intervention if args.intervention else wandb.run.name
wandb.save("params.txt")


DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

from functools import partial

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )

@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

import random

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

dataset = SocialDataset(
            processor=processor,
            intervention = args.intervention,
            are_classifier_prompts = False,
            entity = args.entity,
            batch_size = args.batch_size
        )

image_dump_dir = f'/local/hbansal/minidalle/output_images/{args.output_fname}'
os.makedirs(image_dump_dir, exist_ok=True)

# tokenized_prompts = processor(prompts)
# tokenized_prompt = replicate(tokenized_prompts)

# number of predictions per prompt
n_predictions = 9

# We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm import trange

# generate images
print(jax.device_count())
images = []
for i in trange(max(n_predictions // jax.device_count(), 1)):
    # get a new key
    key, subkey = jax.random.split(key)
    times = len(dataset)//args.batch_size if len(dataset)%args.batch_size==0 else len(dataset)//args.batch_size + 1
    for j in range(times):
        inp = dataset[j]
        tokenized_prompts = inp['tokenized']
        tokenized_prompt = replicate(tokenized_prompts)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for index in range(len(decoded_images)):
            img = Image.fromarray(np.asarray(decoded_images[index] * 255, dtype=np.uint8))
            caption = inp['captions'][index]
            caption_dir = os.path.join(image_dump_dir, caption)
            os.makedirs(caption_dir, exist_ok = True)

            fname = f"{i}.jpg"
            out_fname = os.path.join(caption_dir, fname)
            img.save(out_fname)
wandb.finish()
