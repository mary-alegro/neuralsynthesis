from src.data import create_dataset
from src.models import create_model
from types import SimpleNamespace
from src.util import util
import matplotlib.pyplot as plt
import os

import numpy as np
from skimage.morphology import closing, disk
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte
import pickle
import cv2

class GAN:

    def __init__(self, checkpoints_dir,data_root, model, epoch = 10):
        # mimicks the argparse options Namespace structure
        self.opt = SimpleNamespace()
        self.opt.aspect_ratio = 1.0
        self.opt.batch_size = 1
        self.opt.checkpoints_dir = checkpoints_dir
        self.opt.crop_size = 256
        self.opt.dataroot = data_root
        self.opt.dataset_mode = 'aligned'
        self.opt.direction = 'AtoB'
        self.opt.display_id = -1
        self.opt.display_winsize = 256
        self.opt.epoch = 'latest'
        self.opt.eval = False
        self.opt.force_test_output = '../results/' + model
        self.opt.gpu_ids = []
        self.opt.init_gain = 0.02
        self.opt.init_type = 'normal'
        self.opt.input_nc = 3
        self.opt.isTrain = False
        self.opt.load_iter = epoch
        self.opt.load_size = 256
        self.opt.max_dataset_size = float('inf')
        self.opt.model = model
        self.opt.n_layers_D = 3
        self.opt.name = model
        self.opt.ndf = 64
        self.opt.netD = 'basic'
        self.opt.netG = 'unet_256'
        self.opt.ngf = 64
        self.opt.no_dropout = False
        self.opt.no_flip = True
        self.opt.norm = 'batch'
        self.opt.ntest = float('inf')
        self.opt.num_test = 50
        self.opt.num_threads = 0
        self.opt.output_nc = 3
        self.opt.phase = 'test'
        self.opt.preprocess = 'resize_and_crop'
        self.opt.results_dir = './results/'
        self.opt.serial_batches = True
        self.opt.suffix = ''
        self.opt.verbose = False

        self.opt.num_threads = 0  # test code only supports num_threads = 1
        self.opt.batch_size = 1  # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    def predict(self):
        dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(self.opt)      # create a model given opt.model and other options
        model.setup(self.opt)    # regular setup: load and print networks; create schedulers

        img_rec = None

        if self.opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results

            im = visuals['fake_B']
            img_rec = util.tensor2im(im)

        return img_rec


class SVM:

    def __init__(self,checkpoints_dir):
        model_file = os.path.join(checkpoints_dir,'svm_rbf_model.pickle')
        with open(model_file, 'rb') as input_file:
            self.rbf_kernel_svm = pickle.load(input_file)
        self.se = disk(10)

    #this method could be much simpler but I'm reusing code from a CNN
    def extract_patches(self, full_imgs, patch_h, patch_w, stride_h, stride_w):
        img_h = full_imgs.shape[2]  # height of the full image
        img_w = full_imgs.shape[3]  # width of the full image
        N_patches_img = ((img_h - patch_h) // stride_h + 1) * (
                    (img_w - patch_w) // stride_w + 1)
        N_patches_tot = N_patches_img * full_imgs.shape[0]
        patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
        iter_tot = 0  # iter over the total number of patches (N_patches)

        for i in range(full_imgs.shape[0]):  # loop over the full images
            for h in range((img_h - patch_h) // stride_h + 1):
                for w in range((img_w - patch_w) // stride_w + 1):
                    patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h,
                            w * stride_w:(w * stride_w) + patch_w]
                    patches[iter_tot] = patch
                    iter_tot += 1  # total
        return patches  # array with all the full_imgs divided in patches

    def recompose(self, preds, img_h, img_w, stride_h, stride_w):
        patch_h = preds.shape[2]
        patch_w = preds.shape[3]
        N_patches_h = (img_h - patch_h) // stride_h + 1
        N_patches_w = (img_w - patch_w) // stride_w + 1
        N_patches_img = N_patches_h * N_patches_w

        N_full_imgs = preds.shape[0] // N_patches_img
        full_prob = np.zeros(
            (N_full_imgs, preds.shape[1], img_h, img_w))  # itialize to zero mega array with sum of Probabilities
        full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

        k = 0  # iterator over all the patches
        for i in range(N_full_imgs):
            for h in range((img_h - patch_h) // stride_h + 1):
                for w in range((img_w - patch_w) // stride_w + 1):
                    full_prob[i, :, h * stride_h:(h * stride_h) +
                                                 patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[k]
                    full_sum[i, :, h * stride_h:(h * stride_h) +
                                                patch_h, w * stride_w:(w * stride_w) + patch_w] += 1
                    k += 1
        final_avg = full_prob / full_sum
        print(final_avg.shape)

        return final_avg

    def load_img(self,img_file):
        img_orig = io.imread(img_file)
        if img_orig.shape[1] > 256:
            img_orig = img_orig[:,0:256,:]

        img_orig_gry = img_as_ubyte(rgb2gray(img_orig))
        return img_orig, img_orig_gry

    def extract_texture(self,img_orig_gry):
        img_orig_gry2 = img_orig_gry.reshape((img_orig_gry.shape[0],img_orig_gry.shape[1],1,1))
        img_orig_gry2 = np.transpose(img_orig_gry2,axes=(3,2,0,1))
        patches_tmp = self.extract_patches(img_orig_gry2,30, 30, 10, 10)
        patches_orig = np.transpose(patches_tmp, axes=(0, 2, 3, 1))
        patches_orig = patches_orig.reshape((patches_orig.shape[0], patches_orig.shape[1], patches_orig.shape[2]))
        patches_orig = patches_orig.astype('uint8')
        nPatches = patches_orig.shape[0]

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

    def create_overlay(self,img, mask):
        mask_rgb = np.zeros((mask.shape[0],mask.shape[1],3))
        mask_rgb[...,0] = mask
        over_image = cv2.addWeighted(img, 0.7, mask_rgb.astype('uint8'), 0.3, 0)
        return over_image

    def segment_img(self,img_gry,img_orig):
        patches_orig = self.extract_texture(img_gry)
        features = np.mean(patches_orig, axis=(1, 2))
        predicts = self.rbf_kernel_svm.predict(features)
        mask_patches = np.ones((features.shape[0],30,30))
        for i in range(mask_patches.shape[0]):
            mask_patches[i,...] = mask_patches[i,...] * predicts[i]
        mask_patches[mask_patches > 0] = 255

        mask_patches = mask_patches.reshape((mask_patches.shape[0],1,
                                             mask_patches.shape[1],mask_patches.shape[2]))
        mask_pred = self.recompose(mask_patches,256,256,10,10)
        mask_pred = mask_pred.reshape((mask_pred.shape[2],mask_pred.shape[3]))
        mask = np.zeros(mask_pred.shape)
        mask[mask_pred > 50] = 255
        #postprocess with morphology
        mask_pos = closing(mask,self.se)


        #segment image
        seg_img = img_orig.copy()
        seg_rows,seg_cols = np.nonzero(mask_pos > 0)
        seg_img[seg_rows,seg_cols,0] = 0
        seg_img[seg_rows, seg_cols, 1] = 0
        seg_img[seg_rows, seg_cols, 2] = 0

        return seg_img, mask_pos

    def segment(self, img_file):
        img_orig, img_orig_gry = self.load_img(img_file)
        seg_img, mask = self.segment_img(img_orig_gry, img_orig)
        over_img = self.create_overlay(img_orig,mask)

        return img_orig, seg_img, mask, over_img

class Pipeline:
    def __init__(self, checkpoints_dir,data_root, model, epoch = 10):
        self.gan = GAN(checkpoints_dir,data_root, model, epoch)
        self.svm = SVM(checkpoints_dir)

    def run_pipeline(self,do_seg = False):
        over_img = []
        if do_seg:
            orig_img, seg_img, mask, over_img = self.svm.segment('./db/test/tmp.png')
            gan_img = self.gan.predict()
            rec_img = orig_img.copy()
            rec_img[mask > 0] = gan_img[mask > 0]
        else:
            rec_img = self.gan.predict()

            print(rec_img)
            print(over_img)

        return rec_img,over_img



# test
if __name__ == '__main__':
    # test_img = ''
#     # gan = GAN('./checkpoints','./db','pix2pixpl',10)
#     # img = gan.predict()
#     # plt.imshow(img)
    model_file = '/Users/maryana/Posdoc/Insight/project/data/bedroom/svm/svm_rbf_model.pickle'
    img_file = './db/test/tmp.png'

    svm = SVM('./checkpoints')
    img_orig, img_orig_gry = svm.load_img(img_file)
    seg_img, mask = svm.segment_img(img_orig_gry,img_orig)
    over_img = svm.create_overlay(img_orig,mask)
    plt.imshow(mask)
