import numpy as np
import glob
import os
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage import  img_as_ubyte
from skimage.filters import threshold_otsu
import argparse


def compute_texture(data_dir):

    print('Processing images in folder: {}'.format(data_dir))
    #data_dir = '/Users/maryana/Posdoc/Insight/project/data/bedroom/svm'
    files = glob.glob(os.path.join(data_dir,'*.jpg'))
    for file in files:

        print('File {}'.format(file))

        img = io.imread(file)
        img_orig = img[:,0:256,:]
        img_gt = img[:,256:512,:]

        img_orig_gry = img_as_ubyte(rgb2gray(img_orig))
        img_gt_gry = img_as_ubyte(rgb2gray(img_gt))

        patches_orig = extract_patches_2d(img_orig_gry,(30,30))
        patches_gt = extract_patches_2d(img_gt_gry, (30, 30))
        nPatches = patches_orig.shape[0]

        #feat_names = ['dissimilarity','energy', 'homogeneity','contrast','correlation']
        feat_names = ['dissimilarity', 'energy']
        nFeat = len(feat_names)
        features_orig = np.zeros((nPatches,30,30,nFeat))
        features_gt = np.zeros((nPatches, 30,30, nFeat))

        print('Computing texture')
        for p in range(nPatches):
            patch_orig = patches_orig[p,...]
            patch_gt = patches_gt[p,...]

            glcm_orig = greycomatrix(patch_orig, distances=[5], angles=[0], levels=256,symmetric=True, normed=True)
            glcm_gt = greycomatrix(patch_gt, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

            for f in range(nFeat):
                feat = feat_names[f]
                features_orig[p, :, :,f]  = greycoprops(glcm_orig, feat)[0, 0]
                features_gt[p, :, :,f] = greycoprops(glcm_gt, feat)[0, 0]

        #save texture features
        np.save(file+'_orig_texture.npy',features_orig)
        np.save(file + '_gt_texture.npy', features_gt)

        mean_diff_map = np.zeros(img_gt_gry.shape)

        print('Computing difference maps.')
        for select in range(nFeat):
            p_orig = features_orig[...,select]
            img_orig_gry2 = reconstruct_from_patches_2d(p_orig,img_orig_gry.shape)
            p_gt = features_gt[...,select]
            img_gt_gry2 = reconstruct_from_patches_2d(p_gt, img_gt_gry.shape)
            diff_map = abs(img_orig_gry2 - img_gt_gry2)
            diff_map /= diff_map.max()
            mean_diff_map += diff_map
        mean_diff_map /= nFeat

        np.save(file + '_gt_diffmap.npy', mean_diff_map)

        #threshold
        print('Creating mask.')
        thres = threshold_otsu(mean_diff_map)
        mask = np.zeros(img_gt_gry.shape,dtype='uint8')
        mask[mean_diff_map > thres] = 255
        io.imsave(file + '_mask.png',mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("imagesdir", type=str,
                        help="full path to directory with all results")
    args = parser.parse_args()

    compute_texture(args.imagesdir)








