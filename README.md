# texttoimage
Stable Diffusion Image Generation
This project uses the Stable Diffusion model from the diffusers library and the GPT-2 model from the transformers library to generate images based on textual prompts. The code initializes the models, sets up configuration parameters, and includes a function to generate images.

Requirements
Python 3.6+
PyTorch
diffusers
transformers
tqdm
pandas
numpy
matplotlib
opencv-python
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/stable-diffusion-image-gen.git
cd stable-diffusion-image-gen
Install the required packages:

bash
Copy code
pip install --upgrade diffusers transformers tqdm pandas numpy matplotlib opencv-python
Configuration
The CFG class defines the configuration parameters for the project:

device: The device to run the model on (cuda if available, otherwise cpu).
seed: The random seed for reproducibility.
generator: A PyTorch generator for reproducibility.
image_gen_steps: The number of inference steps for image generation.
image_gen_model_id: The identifier for the image generation model.
image_gen_size: The size of the generated image.
image_gen_guidance_scale: The guidance scale for the image generation model.
prompt_gen_model_id: The identifier for the prompt generation model.
prompt_dataset_size: The size of the prompt dataset.
prompt_max_length: The maximum length of the prompts.
Usage
Initialize the Stable Diffusion model:

python
Copy code
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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',
    guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)
Define a function to generate images:

python
Copy code
from PIL import Image

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    return image
Generate an image:

python
Copy code
prompt = "two trains colliding"
image = generate_image(prompt, image_gen_model)
image.show()
Example
Here is an example of generating an image with the prompt "two trains colliding":

python
Copy code
!pip install --upgrade diffusers transformers -q
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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',
    guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    return image

generate_image("two trains colliding", image_gen_model)
This will generate and display an image based on the provided prompt.

License
This project is licensed under the MIT License.

Acknowledgements
Hugging Face for the diffusers and transformers libraries.
NVIDIA for providing the CUDA toolkit and drivers.
