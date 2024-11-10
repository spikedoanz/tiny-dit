from math import pi as π
from tinygrad import Tensor


def cosine_beta_schedule(steps:int, s=0.008):
  # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
  def linspace(start, end, n): 
    return Tensor.full(n,start) + Tensor.arange(n) * ((end-start) / (n-1))
  α = ((linspace(0, steps, steps + 1)/steps).cos() / (1+s) * π * 0.5) ** 2
  α = α / α[0] # normalize
  β = 1 - (α[1:] / α[:-1])
  return β.clamp(0.0001, 0.9999)


def apply_noise(x: Tensor, step:int, total_steps:int=50):
  def cumprod (x: Tensor): return x.log().cumsum().exp()
  assert step < total_steps, "curr step is larger than total steps"
  Πα = cumprod((1. - cosine_beta_schedule(total_steps)))
  return Πα.sqrt()[step] * x + (1-Πα).sqrt()[step] * Tensor.randn(x.shape) 


def draw(x: Tensor, path="sample.png"):
  from PIL import Image
  import numpy as np
  x = (x - x.min()) / (x.max() - x.min()) * 255
  if len(x.shape) == 3 and x.shape[0] == 1: 
    x = x.squeeze(0)
  x = x.numpy().astype(np.uint8)
  Image.fromarray(x).save(path)


if __name__ == "__main__":
  from tinygrad.nn.datasets import mnist
  X_train, Y_train, X_test, Y_test = mnist()
  normalized_sample = (X_train[0] / 127.5) - 1
  noised_sample     = apply_noise(normalized_sample, 10)
  draw(noised_sample) 
