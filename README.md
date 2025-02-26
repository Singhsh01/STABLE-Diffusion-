# STABLE-Diffusion-

This is a PyTorch implementation of Stable Diffusion, an AI-based model designed for generating high-quality images from text descriptions. This implementation is built from scratch, offering flexibility to modify and experiment with different model configurations and hyperparameters.

# Prerequisites

Before using the model, make sure you have the following software installed:
Python 3.7+
PyTorch (>= 1.9)
CUDA (if you're using a GPU)
Git (for version control)
Install dependencies with:
pip install -r requirements.txt

# Setup

1. Download Weights and Tokenizer Files
To run the Stable Diffusion model, you'll need the following files:
Tokenizer Files
Download vocab.json and merges.txt from the following URL:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer
Save these files in the data folder of the repository.
Model Weights
Download the v1-5-pruned-emaonly.ckpt checkpoint from:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main
Save it in the data folder of the repository.

Ensure that your directory structure looks like this:
/your_project_directory/
    ├── data/
    │   ├── vocab.json
    │   ├── merges.txt
    │   └── v1-5-pruned-emaonly.ckpt
    ├── your_code_files.py
    ├── requirements.txt
    └── README.md

# 2. Tested Fine-Tuned Models

In addition to the base model, you can use various fine-tuned versions of Stable Diffusion. These fine-tuned models have been trained to specialize in specific styles or types of images.

InkPunk Diffusion: https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main
A model fine-tuned for creating "InkPunk" style art.

Link to the model  https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main

Illustration Diffusion: A model fine-tuned on a dataset of illustrations by Hollie Mengert.
Link to the model  https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main

To use any of the fine-tuned models, simply download the respective .ckpt files and save them in the data folder. You can load these checkpoint files into the model to generate specialized outputs.

# 3. Run the Model
Once you've set up the model and downloaded the necessary files, you can run the script to generate images from textual prompts. You can modify the configuration and parameters as needed, depending on the quality and style of images you wish to generate.
Here's an example of how to run the model:

python generate_images.py --prompt "A beautiful landscape with mountains and a river" --output_path "output/image.png"

Make sure that you provide the correct arguments, such as the --prompt for the text input and --output_path for where the generated image should be saved.

# Thanks to 

We would like to acknowledge the contributions and hard work of the following repositories, which have played a significant role in the development of this project:

[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion/)  – The official implementation of Stable Diffusion, which served as the foundation for this project.

[divamgupta/stable-diffusion-tensorflow](https://github.com/divamgupta/stable-diffusion-tensorflow) – A TensorFlow implementation that helped inform the PyTorch implementation.

[kjsman/stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch) – Another PyTorch implementation that contributed ideas and structure to this repository.

[huggingface/diffusers](https://github.com/huggingface/diffusers/) – Hugging Face's diffusers library, which provides tools for easy model deployment and usage.

These repositories have significantly helped with the development and fine-tuning of models for Stable Diffusion and diffusion-based generative models.
