import skimage
from skimage.metrics import structural_similarity as ssim
import numpy as np


def psnr(original, sample):
  mse = np.square(original-sample).mean()
  if mse == 0:
    return mse
  return 20 * np.log10(255/np.sqrt(mse))


def ssim_(original, sample):
  value ,_ = ssim(original, sample, full=True, win_size=3, data_range=1,channel_axis=-1)
  return value