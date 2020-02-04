"""General-purpose test script for image-to-image translation.

"""
from src.options.test_options import TestOptions
from src.data import create_dataset
from src.models import create_model
from src.util.visualizer import save_images_to_path
import sys


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)    # regular setup: load and print networks; create schedulers

    if opt.force_test_output == '':
        print('--force_test_output must be defined.')
        sys.exit()


    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images_to_path (opt.force_test_output, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    #webpage.save()  # save the HTML
