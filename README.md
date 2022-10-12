# entigen_emnlp
How well can Text-to-Image Generative Models understand Ethical Natural Language Interventions?


# Note
Our codebase is broadly similar to [DallEval](https://github.com/j-min/DallEval). However, we provide inference scripts to run DALL.E-mini and Stable Diffusion too. Additionally, we have the capability to intervene on the prompts.

# Models
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [min-DALL.E](https://github.com/kakaobrain/minDALL-E)
- [DALL.E-mini](https://github.com/borisdayma/dalle-mini)

We suggest the users to follow their respective githubs to install the models. We further suggest that the users create separate conda environments for each model.

The users can follow the [this](https://github.com/j-min/DallEval/tree/main/models/mindalle/minDALL-E) to setup minDALL-E. Our scripts are subset of this codebase.

# TODO

- Inference Colab for each model
- Running scores with clip.py