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

#keys = ['5','6', '26', '39', '87', '104', '136', '139']
#keys=['87','136','139']

#L1 norm
mse = 0
mi = 0
ssim = 0
mae = 0

mse_orig = 0
mi_orig = 0
ssim_orig = 0
mae_orig = 0

keys = metrics.keys()

for key in keys:
    mse_key = metrics[key][0]
    mi_key = metrics[key][1]
    ssim_key = metrics[key][2]
    mae_key = metrics[key][3]

    mse_orig += mse_key[0]
    mse += mse_key[-1]

    mi_orig += mi_key[0]
    mi += mi_key[-1]

    ssim_orig += ssim_key[0]
    ssim += ssim_key[-1]

    mae_orig += mae_key[0]
    mae += mae_key[-1]


mse_orig /= nImages
mi_orig /= nImages
ssim_orig /= nImages
mae_orig /= nImages

mse /= nImages
mi /= nImages
ssim /= nImages
mae /= nImages

print('Mean metrics for L1 norm')
print('Original image: MSE {}, MI {}, SSIM {}, mae {}'.format(mse_orig,mi_orig,ssim_orig, mae_orig))
print('Repaired image: MSE {}, MI {}, SSIM {}, mae {}'.format(mse,mi,ssim, mae))

#Perceptual norm
mse_pl = 0
mi_pl = 0
ssim_pl = 0
mae_pl = 0

for key in keys:
    mse_key = metrics_pl[key][0]
    mi_key = metrics_pl[key][1]
    ssim_key = metrics_pl[key][2]
    mae_key = metrics_pl[key][3]

    mse_pl += mse_key[-1]
    mi_pl += mi_key[-1]
    ssim_pl += ssim_key[-1]
    mae_pl += mae_key[-1]

mse_pl /= nImages
mi_pl /= nImages
ssim_pl /= nImages
mae_pl /= nImages

print('Mean metrics for Perceptual Loss norm')
print('Original image: MSE {}, MI {}, SSIM {}, mae {}'.format(mse_orig,mi_orig,ssim_orig, mae_orig))
print('Repaired image: MSE {}, MI {}, SSIM {}, mae {}'.format(mse_pl,mi_pl,ssim_pl, mae_pl))



