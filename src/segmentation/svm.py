import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte
import pickle
import cv2


class SVM:

    def __init__(self,model_file):
        with open(model_file, 'rb') as input_file:
            self.rbf_kernel_svm = pickle.load(input_file)

    def load_img(self,img_file):

        img_orig = io.imread(img_file)
        if img_orig.shape[1] > 256:
            img_orig = img_orig[:,0:256,:]

        img_orig_gry = img_as_ubyte(rgb2gray(img_orig))
        return img_orig, img_orig_gry

    def extract_texture(self,img_orig_gry):

        patches_orig = extract_patches_2d(img_orig_gry,(30,30))
        nPatches = patches_orig.shape[0]

        #feat_names = ['dissimilarity','energy', 'homogeneity','contrast','correlation']
        feat_names = ['dissimilarity', 'energy']
        nFeat = len(feat_names)
        features_orig = np.zeros((nPatches,30,30,nFeat))

        print('Computing texture')
        for p in range(nPatches):
            patch_orig = patches_orig[p,...]
            glcm_orig = greycomatrix(patch_orig, distances=[5], angles=[0], levels=256,symmetric=True, normed=True)

            for f in range(nFeat):
                feat = feat_names[f]
                features_orig[p, :, :,f] = greycoprops(glcm_orig, feat)[0, 0]

        return features_orig

    def create_overlay(self,img,mask):
        mask_rgb = np.zeros((mask.shape[0],mask.shape[1],3))
        mask_rgb[...,0] = mask
        over_image = cv2.addWeighted(img, 0.7, mask_rgb.astype('uint8'), 0.3, 0)
        return over_image

    def segment_img(self,img_file):
        patches_orig = self.extract_texture(img_file)
        features = np.mean(patches_orig, axis=(1, 2))
        predicts = self.rbf_kernel_svm.predict(features)
        mask_patches = np.ones((features.shape[0],30,30))
        for i in range(mask_patches.shape[0]):
            mask_patches[i,...] = mask_patches[i,...] * predicts[i]
        mask_patches[mask_patches > 0] = 255

        mask_pred = reconstruct_from_patches_2d(mask_patches, (256,256))
        mask = np.zeros(mask_pred.shape)
        mask[mask_pred > 100] = 255

        over_img = self.create_overlay(img_orig,mask)

        #segment image
        seg_img = img_orig.copy()
        seg_rows,seg_cols = np.nonzero(mask > 0)
        seg_img[seg_rows,seg_cols,0] = 0
        seg_img[seg_rows, seg_cols, 1] = 0
        seg_img[seg_rows, seg_cols, 2] = 0

        return seg_img, mask, over_img

if __name__ == '__main__':
    model_file = '/Users/maryana/Posdoc/Insight/project/data/bedroom/svm/svm_rbf_model.pickle'
    img_file = '/db/test/tmp.png'

    svm = SVM(model_file)
    img_orig, img_orig_gry = svm.load_img(img_file)
    seg_img, mask, over_img = svm.segment_img(img_orig_gry)
    plt.imshow(mask)

