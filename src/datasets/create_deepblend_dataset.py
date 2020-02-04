import glob
import os
import argparse
import cv2
import random
import math
import shutil
import skimage.io as io


p2p_dim = (256,256)
prop = (0.8,0.1)

def build_dataset(data_path,out_path):

    #example data_path: /path/to/data_train/Bedroom/pix2pix
    #example out_path /path/to/data_pix2pix/A <--- images to be reconstructed
    #                                       train
    #                                       test
    #                                       val
    #example out_path /path/to/data_pix2pix/B <--- reference images
    #                                       train
    #                                       test
    #                                       val

    print('Creating dataset...')

    directories = glob.glob(os.path.join(data_path,'**/'))

    nDir = len(directories)
    print('{} directories found.'.format(nDir))

    dir_out_A = os.path.join(out_path, 'A')
    dir_out_B = os.path.join(out_path, 'B')
    #create output directories
    if not os.path.exists(dir_out_A):
        os.mkdir(dir_out_A)
    if not os.path.exists(dir_out_B):
        os.mkdir(dir_out_B)

    #out_A = os.path.join(out_path,'A')
    #out_B = os.path.join(out_path,'B')
    #
    # 1st: create pairs
    #
    count = 0
    for direc in directories:
        print('Processing {}'.format(direc))

        dir_p2p = os.path.join(direc, 'pix2pix')
        #get all reference files
        reference = glob.glob(os.path.join(dir_p2p,'*reference.jpg')) #reference filename format: 0000_reference.jpg

        for ref in reference: #iterate over all images related to a reference image
            basename = os.path.basename(ref)
            img_num = basename.split('_')[0]

            #load reference image and reshape (pix2pix input is 256x256)
            #img_ref = cv2.imread(ref,cv2.IMREAD_UNCHANGED)
            img_ref = io.imread(ref)
            img_ref_p2p = cv2.resize(img_ref,p2p_dim)

            channels = glob.glob(os.path.join(dir_p2p,img_num+'_local*'))
            for chan in channels:
                #img_chan = cv2.imread(chan,cv2.IMREAD_UNCHANGED)
                img_chan = io.imread(chan)
                img_chan_p2p = cv2.resize(img_chan, p2p_dim)
                #save pair {AB} (channel,reference), reference will be repeated several times
                #cv2.imwrite(os.path.join(dir_p2p_A, str(count) + '.jpg'), img_chan_p2p)
                #cv2.imwrite(os.path.join(dir_p2p_B,str(count)+'.jpg'),img_ref_p2p)

                io.imsave(os.path.join(dir_out_A, str(count) + '.jpg'), img_chan_p2p)
                io.imsave(os.path.join(dir_out_B,str(count)+'.jpg'),img_ref_p2p)
                count+=1

    #
    # 2nd shuffle and split in train, test, val
    #

    print('Splitting dataset...')

    files_A = glob.glob(os.path.join(dir_out_A,'*.jpg'))
    files_B = glob.glob(os.path.join(dir_out_B, '*.jpg'))
    nA = len(files_A)
    nB = len(files_B)

    #make sure number of files is the same
    if nA != nB:
        print('Error: number of files are different {}/{}. Stopping!'.format(nA,nB))
        return

    print('{} total files.'.format(nA))

    #shuffle array
    random.shuffle(files_A)
    n_train = math.floor(nA*prop[0])
    n_test = math.floor(nA*prop[1])

    print('{} train files.'.format(n_train))
    print('{} test files.'.format(n_test))
    print('{} validation files'.format(nA - (n_test+n_train)))

    #save training set
    if not os.path.exists(os.path.join(dir_out_A,'train')):
        os.mkdir(os.path.join(dir_out_A,'train'))
    if not os.path.exists(os.path.join(dir_out_B, 'train')):
        os.mkdir(os.path.join(dir_out_B, 'train'))
    for ind in range(0,n_train):
        A = files_A[ind]
        basename = os.path.basename(A)
        B = os.path.join(dir_out_B,basename)
        #A and B must be copied together
        new_A = os.path.join(dir_out_A,'train',basename)
        new_B = os.path.join(dir_out_B,'train',basename)
        #shutil.copy(A,new_A)
        shutil.move(A,os.path.join(dir_out_A,'train'))
        #shutil.copy(B,new_B)
        shutil.move(B, os.path.join(dir_out_B,'train'))

    #save test set
    if not os.path.exists(os.path.join(dir_out_A,'test')):
        os.mkdir(os.path.join(dir_out_A, 'test'))
    if not os.path.exists(os.path.join(dir_out_B, 'test')):
        os.mkdir(os.path.join(dir_out_B, 'test'))
    for ind in range(n_train,n_train+n_test):
        A = files_A[ind]
        basename = os.path.basename(A)
        B = os.path.join(dir_out_B,basename)
        #A and B must be copied together
        new_A = os.path.join(dir_out_A,'test',basename)
        new_B = os.path.join(dir_out_B,'test',basename)
        #shutil.copy(A,new_A)
        shutil.move(A,os.path.join(dir_out_A,'test'))
        #shutil.copy(B,new_B)
        shutil.move(B, os.path.join(dir_out_B,'test'))

    #save validation set
    if not os.path.exists(os.path.join(dir_out_A,'val')):
        os.mkdir(os.path.join(dir_out_A,'val'))
    if not os.path.exists(os.path.join(dir_out_B, 'val')):
        os.mkdir(os.path.join(dir_out_B, 'val'))
    for ind in range(n_train+n_test,nA):
        A = files_A[ind]
        basename = os.path.basename(A)
        B = os.path.join(dir_out_B,basename)
        #A and B must be copied together
        new_A = os.path.join(dir_out_A,'val',basename)
        new_B = os.path.join(dir_out_B,'val',basename)
        #shutil.copy(A,new_A)
        shutil.move(A,os.path.join(dir_out_A,'val'))
        #shutil.copy(B,new_B)
        shutil.move(B, os.path.join(dir_out_B,'val'))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=str,
                        help="full path to directory with all data")
    parser.add_argument("output_path", type=str,
                        help="full path to output dataset")

    args = parser.parse_args()
    print(args.data_path,args.output_path)
    build_dataset(args.data_path,args.output_path)

if __name__ == '__main__':
    main()


