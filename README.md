# This is a personal project, for educational purposes only!
# About this project:
1. MaskGIT or Masked Generative Image Transformer is a two-stage image generation model. The first stage is to quantize an image to a sequence of discrete tokens. In the second stage, an autoregressive model is learned to generate image tokens sequentially based on the previously generated result.
2. The first stage (VQGAN) was trained on Celebrity Face Image Dataset (https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) with minimal settings (image size: 32, depth: 3, latent dim: 64,...).
> See the "results" folder for more details. \
     ![image](https://github.com/tomsawyer0224/maskgit/assets/130035084/3b00811f-1fb4-471b-a998-18b3d6ed9e25) \
     ![image](https://github.com/tomsawyer0224/maskgit/assets/130035084/619dcad6-67fe-4ba3-b474-b5b32334b113)
4. The second stage (Transformer) has not been trained yet.
# How to use:
1. Clone this repo, cd to maskgit.
2. Install the requirements: pip install -q -r requirements.txt.
3. Traning the VQGAN: modify the config file (configs/vqgan.yaml), then run the below command:
```
 python train.py \
   --phase "vqgan" \
   --config_file "./configs/vqgan.yaml" \
   --max_epochs 10 \
   --ckpt_path "path/to/checkpoint" # when you want to resume training
```
5. Training the Transformer: modify the config file (configs/transformer.yaml, in this phase you must provide vqgan checkpoint path), then run the below command:
```
 python train.py \
   --phase "transformer" \
   --config_file "./configs/transformer.yaml" \
   --max_epochs 10 \
   --ckpt_path "path/to/checkpoint" # when you want to resume training
```
> The logs and checkpoints will be saved to the "results" folder. \
Note: This project was built on Google Colab, it may not work on other platforms.
# Based on:
  https://arxiv.org/abs/2202.04200 \
  https://arxiv.org/abs/2012.09841 \
  https://github.com/dome272/MaskGIT-pytorch \
  https://github.com/CompVis/taming-transformers \
  https://github.com/google-research/maskgit \
  https://github.com/hmorimitsu/maskgit-torch \
  https://github.com/richzhang/PerceptualSimilarity \
  https://github.com/lucidrains/vector-quantize-pytorch \
  https://huggingface.co/docs/diffusers/api/models/unet2d
