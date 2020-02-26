import skimage.io as io
import os
import glob
import sys
import argparse

mask_dir = ''
results_dir = ''

def apply_seg(results_dir,mask_dir):

    folders = glob.glob(os.path.join(results_dir,'**/'))
    folders.sort(key=lambda x: float(x.strip('/').split('_')[-1]))

    print('Parsing folders...')
    for folder in folders: #epoch folder array is ordered
        print('Parsing {}'.format(folder))
        basename = os.path.basename(os.path.normpath(folder))
        epoch = int(basename.split('_')[-1])
        images = glob.glob(os.path.join(folder,'*_fake_B.png')) #get all images generated by the GAN generator

        for im_fake in images:
            im_basename = os.path.basename(im_fake)
            basedir = os.path.dirname(im_fake)
            scene_id = im_basename.split('_fake_B')[0]
            img_gt = os.path.join(basedir,scene_id+'_real_B.png')
            img_orig = os.path.join(basedir, scene_id + '_real_A.png')

            mask_file = os.path.join(mask_dir,scene_id+'.jpg_mask.png')

            if not os.path.exists(img_gt):
                print('Warning: ground truth image {} does not exist.'.format(img_gt))
            if not os.path.exists(img_orig):
                print('Warning: original image {} does not exist.'.format(img_orig))
            if not os.path.exists(mask_file):
                print('Warning: mask image {} does not exist.Skipping.'.format(mask_file))
                continue

            mask = io.imread(mask_file)
            orig = io.imread(img_orig)
            fake_B = io.imread(im_fake)

            seg_img = orig.copy()
            seg_img[mask == 255] = fake_B[mask == 255]
            seg_name = os.path.join(basedir,scene_id+'_fake_B_seg.png')
            print('Saving {}'.format(seg_name))
            io.imsave(seg_name,seg_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("resultsroot", type=str,
                        help="full path to directory with all results")
    parser.add_argument("maskdir", type=str,
                        help="full path to mask folder)")



    args = parser.parse_args()
    apply_seg(args.resultsroot, args.maskdir)









