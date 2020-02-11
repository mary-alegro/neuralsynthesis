"""General-purpose test script for image-to-image translation.

"""
from src.options.test_options import TestOptions
from src.data import create_dataset
from src.models import create_model
from src.util.visualizer import save_images_to_path
import sys
from types import SimpleNamespace
from src.util import util
import os
import matplotlib.pyplot as plt


def predict(checkpoints_dir,data_root, model, epoch = 10):

    opt = SimpleNamespace()
    opt.aspect_ratio = 1.0
    opt.batch_size = 1
    #opt.checkpoints_dir = '../checkpoints'
    opt.checkpoints_dir = checkpoints_dir
    opt.crop_size = 256
    #opt.dataroot = '/Users/maryana/Posdoc/Insight/data/bedroom'
    opt.dataroot = data_root
    opt.dataset_mode = 'aligned'
    opt.direction = 'AtoB'
    opt.display_id = -1
    opt.display_winsize = 256
    opt.epoch = 'latest'
    opt.eval = False
    opt.force_test_output = '../results/'+ model
    opt.gpu_ids = []
    opt.init_gain = 0.02
    opt.init_type = 'normal'
    opt.input_nc = 3
    opt.isTrain = False
    opt.load_iter = epoch
    opt.load_size = 256
    opt.max_dataset_size = float('inf')
    #opt.model = 'pix2pix'
    opt.model = model
    opt.n_layers_D = 3
    #opt.name = 'bedroom'
    opt.name = model
    opt.ndf = 64
    opt.netD = 'basic'
    opt.netG = 'unet_256'
    opt.ngf = 64
    opt.no_dropout = False
    opt.no_flip = True
    opt.norm = 'batch'
    opt.ntest = float('inf')
    opt.num_test = 50
    opt.num_threads = 0
    opt.output_nc = 3
    opt.phase = 'test'
    opt.preprocess = 'resize_and_crop'
    opt.results_dir = './results/'
    opt.serial_batches = True
    opt.suffix = ''
    opt.verbose = False

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)    # regular setup: load and print networks; create schedulers

    img_rec = None

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        im = visuals['fake_B']
        img_rec = util.tensor2im(im)


    return img_rec


if __name__ == '__main__':
    test_img = ''

    img = predict('./checkpoints','./db','pix2pixpl',10)
    plt.imshow(img)
