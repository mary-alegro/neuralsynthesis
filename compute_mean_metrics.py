import glob
import argparse
import os
import sys
import cv2
from src.util.scores import *
import pickle
from skimage.measure import compare_ssim


with open('/Users/maryana/Posdoc/Insight/project/results/bedroom/metrics.pickle','rb') as input_file:
    metrics = pickle.load(input_file)

with open('/Users/maryana/Posdoc/Insight/project/results/bedroom_pl/metrics.pickle','rb') as input_file:
    metrics_pl = pickle.load(input_file)

nImages = len(metrics.keys())
nImages_pl = len(metrics_pl.keys())

#L1 norm
mse = 0
mi = 0
ssim = 0

mse_orig = 0
mi_orig = 0
ssim_orig = 0

keys = ['5','6', '26', '39', '87', '104', '136', '139']

for key in keys:
    mse_key = metrics[key][0]
    mi_key = metrics[key][1]
    ssim_key = metrics[key][2]

    mse_orig += mse_key[0]
    mse += mse_key[-1]
    mi_orig += mi_key[0]
    mi += mi_key[-1]
    ssim_orig += ssim_key[0]
    ssim += ssim_key[-1]

mse_orig /= nImages
mi_orig /= nImages
ssim_orig /= nImages

mse /= nImages
mi /= nImages
ssim /= nImages

print('Mean metrics for L1 norm')
print('Original image: MSE {}, MI {}, SSIM {}'.format(mse_orig,mi_orig,ssim_orig))
print('Repaired image: MSE {}, MI {}, SSIM {}'.format(mse,mi,ssim))



