from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import glob
import os
import argparse
import random
import numpy as np
import skimage.io as io
from sklearn.feature_extraction.image import extract_patches_2d

def rnd_list(start,end, n):
    arr = [random.randint(start,end) for _ in range(n)]
    return arr

def create_feature_arr(featdir,file_arr):
    features = []
    classes = []
    for textr_file in file_arr:
        basename = os.path.basename(textr_file)
        idx = basename.find('_orig')
        img_name = basename[0:idx]
        mask_file = os.path.join(featdir,img_name+'_mask.png')

        texture = np.load(textr_file) #texture arr format is [p,30,30,2] where
        # p is the number of patches in the image and 2 is the number  of texture features
        mask = io.imread(mask_file)
        mask_patches = extract_patches_2d(mask,(30,30)) #break mask into patches

        X = np.mean(texture,axis=(1,2))
        y = np.mean(mask_patches,axis=(1,2))
        y[y < 100] = 0 #make sure y is [0,1]
        y[y > 0] = 1

        ### balance data ###
        idx_fore = np.nonzero(y == 1)[0]
        nFore = len(idx_fore)
        idx_back = np.nonzero(y  == 0)[0]
        nBack = len(idx_back)

        if nBack > nFore:
            rnd_idx = rnd_list(0,nBack-1,nFore)
            new_idx_back = idx_back[rnd_idx] #randomly select nFore indices from background patches
            X_back = X[new_idx_back,...]
            X_fore = X[idx_fore]

            y_back = y[new_idx_back]
            y_fore = y[idx_fore]

            X = np.concatenate((X_back,X_fore),axis=0)
            y = np.concatenate((y_back,y_fore),axis=0)

        elif nBack < nFore:
            rnd_idx = rnd_list(0,nFore-1,nBack)
            new_idx_fore = idx_fore[rnd_idx] #randomly select nFore indices from background patches
            X_back = X[idx_back,...]
            X_fore = X[new_idx_fore]

            y_back = y[idx_back]
            y_fore = y[new_idx_fore]

            X = np.concatenate((X_back,X_fore),axis=0)
            y = np.concatenate((y_back,y_fore),axis=0)

        #shuffle final feature and classes array
        nSamples = X.shape[0]
        new_idx = np.arange(nSamples)
        np.random.shuffle(new_idx)
        X = X[new_idx,...]
        y = y[new_idx]

        if features == [] and classes == []:
            features = X
            classes = y
        else:
            features = np.concatenate((features,X), axis=0)
            classes = np.concatenate((classes,y), axis=0)

    return features,classes


def train_svm(featdir,outdir):
    feat_files = glob.glob(os.path.join(featdir,'*_orig_texture.npy'))
    nFiles = len(feat_files)
    nTrain = round(0.8*nFiles)
    nTest = nFiles - nTrain
    random.shuffle(feat_files)

    train_files = [feat_files[i] for i in range(nTrain)]
    test_files = [feat_files[i] for i in range(nTrain,nTrain+nTest)]

    #tmp_ids = [10,15,16,17,21,22,26,29,105]
    tmp_str = '/Users/maryana/Posdoc/Insight/project/data/bedroom/svm/{}.jpg_orig_texture.npy'
    #train_files = [tmp_str.format(i) for i in tmp_ids]


    train_features,train_classes = create_feature_arr(featdir,train_files)

    rbf_kernel_svm = Pipeline((('scaler',StandardScaler()),('svm_clf',SVC(kernel='rbf', gamma=5, C=0.001))))
    rbf_kernel_svm.fit(train_features,train_classes)

    test_file = '/Users/maryana/Posdoc/Insight/project/data/bedroom/svm/104.jpg_orig_texture.npy'
    img_file = '/Users/maryana/Posdoc/Insight/project/data/bedroom/svm/104.jpg'
    mask_file = '/Users/maryana/Posdoc/Insight/project/data/bedroom/svm/104.jpg_mask.png'

    test_textr = np.load(test_file)
    test_textr = np.mean(test_textr,axis=(1,2))
    results = rbf_kernel_svm.predict(test_textr)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("festuresdir", type=str, help="full path to directory with all results")
    parser.add_argument("outdir", type=str, help="full path to directory with all results")
    args = parser.parse_args()

    train_svm(args.festuresdir, args.outdir)


