from src.util.scores import *
from skimage.measure import compare_ssim
from skimage.color import rgb2gray
from skimage import img_as_ubyte



class MetricsHelper:
    def __init__(self, img_orig,img_rec,img_gt):
        self.img_gt_gry = img_as_ubyte(rgb2gray(img_gt))
        self.img_rec_gry = img_as_ubyte(rgb2gray(img_rec))
        self.img_orig_gry = img_as_ubyte(rgb2gray(img_orig))

    def compute_mse(self):
        mse_orig_gt = mse(self.img_gt_gry, self.img_orig_gry)
        mse_rec_gt = mse(self.img_gt_gry, self.img_rec_gry)
        return mse_orig_gt,mse_rec_gt

    def compute_mae(self):
        mae_orig_gt = mae(self.img_gt_gry, self.img_orig_gry)
        mae_rec_gt = mae(self.img_gt_gry, self.img_rec_gry)
        return mae_orig_gt,mae_rec_gt

    def compute_ssim(self):
        ssim_orig_gt, diff = compare_ssim(self.img_orig_gry, self.img_gt_gry, full=True)
        ssim_rec_gt, diff = compare_ssim(self.img_rec_gry, self.img_gt_gry, full=True)
        return ssim_orig_gt,ssim_rec_gt


