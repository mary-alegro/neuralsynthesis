import glob
import argparse
import os
import sys
import cv2
from src.util.scores import *
import pickle
from skimage.measure import compare_ssim

DEFAULT_W = 256
DEFAULT_H = 256

def compute_metrics(args):

    results_dir = args.resultsroot
    output_dir = args.outputdir
    nMetrics = 0
    if args.mse:
        nMetrics += 1
    if args.mi:
        nMetrics += 1
    if args.ssim:
        nMetrics += 1

    if nMetrics == 0:
        print('Nothing to compute.')
        sys.exist()

    folders = glob.glob(os.path.join(results_dir,'**/'))
    #sort folders
    folders.sort(key=lambda x: float(x.strip('/').split('_')[-1]))
    isFirst = True
    metric_files = {}

    print('Parsing folders...')
    for folder in folders: #epoch folder array is ordered
        print('Parsing {}'.format(folder))
        basename = os.path.basename(os.path.normpath(folder))
        epoch = int(basename.split('_')[-1])

        # get all images generated by the GAN generator
        # that underwent the pre-segmentation step
        images = glob.glob(os.path.join(folder,'*_fake_B_seg.png'))

        for im_fake in images:
            im_basename = os.path.basename(im_fake)
            basedir = os.path.dirname(im_fake)
            scene_id = im_basename.split('_fake_B')[0]
            img_gt = os.path.join(basedir,scene_id+'_real_B.png')
            img_orig = os.path.join(basedir, scene_id + '_real_A.png')

            if not os.path.exists(img_gt):
                print('Warning: ground truth image {} does not exist'.format(img_gt))
            if not os.path.exists(img_orig):
                print('Warning: original image {} does not exist'.format(img_orig))

            if isFirst:
                metric_files[scene_id] = []
                metric_files[scene_id].append((epoch,im_fake,img_orig,img_gt))
            else:
                if scene_id in metric_files:
                    metric_files[scene_id].append((epoch,im_fake,img_orig,img_gt))
                else:
                    print('Warning: epoch {} missing for file {}'.format(epoch,im_fake))

        isFirst = False

    print('Computing scores...')
    metrics = {}
    output_file = os.path.join(output_dir, 'metrics_seg.pickle')
    for scene in metric_files.keys(): #each keys is a different scene

        print('Metrics for scene {}'.format(scene))

        file_array = metric_files[scene]
        nFiles = len(file_array)

        scores = np.zeros((nMetrics,nFiles+1)) #(row = metric, col = orig + nEpochs) scores for each scene
        #compute metrics for epochs
        count = 1
        for fi_tuple in file_array: #iterates over epochs
            im_fake = fi_tuple[1]
            im_orig = fi_tuple[2]
            im_gt = fi_tuple[3]

            img_fake = cv2.imread(im_fake)
            img_orig = cv2.imread(im_orig)
            img_gt = cv2.imread(im_gt)

            im_fake_gry = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)
            im_orig_gry = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            im_gt_gry = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)

            ch1_orig = img_orig[...,0]
            ch2_orig = img_orig[...,1]
            ch3_orig = img_orig[...,2]

            ch1_rec = img_fake[...,0] #reconstructed (aka fake) image
            ch2_rec = img_fake[...,1]
            ch3_rec = img_fake[...,2]

            ch1_gt = img_gt[...,0]
            ch2_gt = img_gt[...,1]
            ch3_gt = img_gt[...,2]

            # compute metrics for original image
            if args.mse:
                #mse_orig_gt = (mse(ch1_orig, ch1_gt) + mse(ch2_orig, ch2_gt) + mse(ch3_orig, ch3_gt)) / 3
                #mse_rec_gt = (mse(ch1_rec, ch1_gt) + mse(ch2_rec, ch2_gt) + mse(ch3_rec, ch3_gt)) / 3

                mse_orig_gt = mse(im_gt_gry,im_orig_gry)
                mse_rec_gt = mse(im_gt_gry,im_fake_gry)

                scores[0, 0] = mse_orig_gt  # original image x GT
                scores[0, count] = mse_rec_gt  # reconstricted image x GT

            if args.mi:
                mi_orig_gt = (mutual_information(ch1_orig, ch1_gt) + mutual_information(ch2_orig, ch2_gt) +
                              mutual_information(ch3_orig, ch3_gt)) / 3
                mi_rec_gt = (mutual_information(ch1_rec, ch1_gt) + mutual_information(ch2_rec, ch2_gt) +
                              mutual_information(ch3_rec, ch3_gt)) / 3
                scores[1, 0] = mi_orig_gt
                scores[1,count] = mi_rec_gt

            if args.ssim:
                ssim_orig_gt,diff = compare_ssim(im_orig_gry,im_gt_gry, full=True)
                ssim_rec_gt,diff = compare_ssim(im_fake_gry, im_gt_gry, full=True)
                scores[2,0] = ssim_orig_gt
                scores[2,count] = ssim_rec_gt

            count += 1

        metrics[scene] = scores

    print('Saving scores to {}'.format(output_file))
    with open(output_file, 'wb') as fp:
        pickle.dump(metrics, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("resultsroot", type=str,
                        help="full path to directory with all results")
    parser.add_argument("outputdir", type=str,
                        help="full path to metrics output dir")
    parser.add_argument("--mse", type=bool, default=True,
                        help="compute ")
    parser.add_argument("--mi", type=int, default=True,
                        help="output width")
    parser.add_argument("--ssim", type=int, default=True,
                        help="output width")


    args = parser.parse_args()
    compute_metrics(args)
