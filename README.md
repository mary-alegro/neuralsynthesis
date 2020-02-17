# Neural Render
This repository contains the code for the Neural Render project, developed during the AI Insight Data Science fellowship.
Neural Render is based on the [pix2pix](https://phillipi.github.io/pix2pix/) network architecture that has had great success performing tasks like style transfer and image restoration. The [slides presentation](https://docs.google.com/presentation/d/16bGAosRl8geZuzGwm6Q3N-gMLayl0CALilGNiwU0zWs/edit#slide=id.g7d5c0ef7ba_0_8952) has more information about Neural Render.

## Requirements
```
pytorch 1.4.0 
pytorch-cpu 1.1.0 
numpy 1.17.3 
opencv 4.2.0   
opencv-python 4.2.0.32 
pillow 7.0.0  
scikit-image 0.16.2 
scikit-learn 0.22.1 
torchfile 0.1.0  
torchvision 0.5.0    
streamlit 0.54.0
```
## Setup and run Neural Render app
First clone repository:
```
clone https://github.com/mary-alegro/neuralsynthesis
```
Enter the repository directory:
```
cd neuralsynthesis
```
Download the [pre-trained weights](https://www.dropbox.com/s/hgutluvc3r2lgwl/neural_render_weights.zip?dl=0)
and unzip the file into the _app/checkpoints_ directory.

Export the `PYTHONPATH` variables:
```
export PYTHONPATH=$PYTHONPATH:../:../src:../app:
```

Run the app:
```
cd app
streamlit run neural_render.py
```
In the app, you can upload files (there are some test files in the _test_imgs/_ folder) and see the results using both the L1 loss and Perceptual loss models. You can also select different epochs and see how the results look like. Note: the image format used by pix2pix [pytorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), on which this project is based, requires that the input (rendered) image and ground truth image are resized to 256x256 pixels and concatenated together. 


![neural render](./assets/app.gif)

## Create a dataset
First, download and uncompress the raw data from [DeepBlending](https://repo-sam.inria.fr/fungraph/deep-blending/data/DeepBlendingTrainingData.zip). NOte it's a 45Gb files.

In this example we will assume the following folder structure:
```
/data/data_train/scenes <- images from DeepBlending
	|
	+ Yellowhouse-12
	+ ...

/data
 |
 + datasets/
 	|
 	+ scene/ <- 'scene' is also the dataset name used by the train and test scripts 
 		|
 		+ test/
 		+ train/
 		+ val/
 |
 + checkpoints/
 	|
 	+ scene/
 		|
 		+ *.pth <- network weights
 |
 + results/
 	|
 	+ scene/
 		|
 		+ scene_N <- results obtained during inference sing weights from epoch N
 		+ ...
```

Copy the files that are going to be used in the dataset to a different folder:
```
cd /data/data_train/secenes/Yellowhouse-12
mkdir pix2pix
cp * cp *_reference.jpg pix2pix
cp *_local_layer_0_colors_sample_0003_path_0000.jpg pix2pix
cp *_local_layer_3_colors_sample_0003_path_0000.jpg pix2pix
```

Use `create_deepblend_dataset.py` to pre-process the split the dataset in train, test and validation 
```
python src/datasets/create_deepblend_dataset.py /data/data_train/scenes /data/datasets/scene
```
This will create folders `A` with all rendered images and `B` with all ground truth images inside `/data/datasets/scene`. 

Now use `combine_A_and_B.py` to create the final dataset. 
```
python src/datasets/combine_A_and_B.py --fold_A /data/datasets/scene/A --fold_B /data/datasets/scene/B --fold_AB /data/datasets/scene
```

Optionally, you can delete folders `A` and `B` . 

## Train
First, start Visdom:
```
 python -m visdom.server -port 6006
```
Train using the Perceptual loss model for 500 epochs:
```
python train_perceptual_loss.py --dataroot /data/datasets/scene --name bedroom_pl --model pix2pixpl --checkpoints_dir /data/checkpoints --display_port 6006 --dataset_mode aligned --n_epochs 250 --n_epochs_decay 250 --display_freq 100
```
Similarly, train using the L1 loss model:
```
python train.py --dataroot /data/datasets/scene --name bedroom_pl --model pix2pix --checkpoints_dir /data/checkpoints --display_port 6006 --dataset_mode aligned --n_epochs 250 --n_epochs_decay 250 --display_freq 100
```
If you don't have a GPU, add option `--gpu_id -1` to the commands above.

Training information can be visualized using the browser, just connect to `localhost:6006`. 

## Test

## Compute image quality metrics
First, compute the metrics:
```
python src/metrics/ 
```
This will compute MSE, MI, MAE and SSIM metrics for each rendered/ground truth image pair and save results into a file named _metrics.pickle_.

Now compute the mean of all metrics:
```
python 
```


