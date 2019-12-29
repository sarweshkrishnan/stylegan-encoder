import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

def save_image(img_array, img_name):
  img = PIL.Image.fromarray(img_array, 'RGB')
  img.save(os.path.join('amplify_expression', f'{img_name}.png'), 'PNG')

def main():
  # Initialize generator
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, 1, randomize_noise=False)

    # load latents
    latent1 = np.load('./dlatent_dir/s007-00_img.npy')
    # latent2 = np.load('./dlatent_dir/s003-01_img.npy')

    latent = latent1.copy()
    latent[:8] = 2 * latent1[:8]

    generated_image = generator.generate_images(latent)
    save_image(generated_image[0], amp_1)

if __name__ == "__main__":
    main()
