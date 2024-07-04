This is a personal project, for educational purpose only!

Based on:
  https://arxiv.org/abs/2202.04200,
  https://arxiv.org/abs/2012.09841,
  https://github.com/dome272/MaskGIT-pytorch,
  https://github.com/CompVis/taming-transformers,
  https://github.com/google-research/maskgit,
  https://github.com/hmorimitsu/maskgit-torch,
  https://github.com/richzhang/PerceptualSimilarity,
  https://github.com/lucidrains/vector-quantize-pytorch,
  https://huggingface.co/docs/diffusers/api/models/unet2d
  
About this project:
  1. Maskgit is a two-stages image generation model (the first stage is to quantize an image to a sequence of discrete tokens. In the second stage, an autoregressive model is learned to generate image tokens sequentially based on the previously generated result.
  2. First stage (VQGAN) was trained on Celebrity Face Image Dataset (https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) with minimal settings (image size: 32, depth: 3, latent dim: 64,...). Please see the "results" folder for more informations.
     ![image](https://github.com/tomsawyer0224/maskgit/assets/130035084/3b00811f-1fb4-471b-a998-18b3d6ed9e25)
     ![image](https://github.com/tomsawyer0224/maskgit/assets/130035084/619dcad6-67fe-4ba3-b474-b5b32334b113)
  3. Second stage (Transformer) has not been trained yet because I don't have enough computational resources (Google Colab free tier).
