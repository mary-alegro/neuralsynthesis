# Neural Render
This repository contains the code for the Neural Render project, developed during the AI Insight Data Science fellowship.
Neural Render is based on the [pix2pix](https://phillipi.github.io/pix2pix/) network architecture that has had great success performing tasks like style transfer and image restoration. The [presentation slides](https://docs.google.com/presentation/d/16bGAosRl8geZuzGwm6Q3N-gMLayl0CALilGNiwU0zWs/edit#slide=id.g7d5c0ef7ba_0_8952) have more information about Neural Render.

## Requirements and environment setup
Neural Render was tested with:
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
First, create a new environment using _conda_:
```
conda create --name neural_render
conda activate neural_render
```
Install the dependencies using _conda_:
```
conda install -f -y -q --name neural_render -c conda-forge --file requirements_conda.txt
```
Some dependencies must be installed with _pip_: 
```
pip install -r requirements.txt
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
export PYTHONPATH=$PYTHONPATH:../:../src:../app
```
Note that in some systems it might be necessary to use the full path to the cloned repository.

Run the app:
```
cd app
streamlit run neural_render.py
```
In the app, you can upload files (there are some test files in the _test_imgs/_ folder) and see the results using both the L1 loss and Perceptual loss models. You can also select different epochs and see how the results look like. Note: the image format used by pix2pix [pytorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), on which this project is based, requires that the input (rendered) image and ground truth image are resized to 256x256 pixels and concatenated together. 


![neural render](./assets/app.gif)

## Create a dataset
First, download and uncompress the raw data from [DeepBlending](https://repo-sam.inria.fr/fungraph/deep-blending/data/DeepBlendingTrainingData.zip). Note it's a 45Gb file.

In the next sections we will assume the following folder structure:
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
cd /data/data_train/scenes/Yellowhouse-12
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
python train_perceptual_loss.py --dataroot /data/datasets/scene --name scene --model pix2pixpl --checkpoints_dir /data/checkpoints --display_port 6006 --dataset_mode aligned --n_epochs 250 --n_epochs_decay 250 --display_freq 100
```
Similarly, train using the L1 loss model:
```
python train.py --dataroot /data/datasets/scene --name scene --model pix2pix --checkpoints_dir /data/checkpoints --display_port 6006 --dataset_mode aligned --n_epochs 250 --n_epochs_decay 250 --display_freq 100
```
Add the `--gpu_id -1` option to use the CPU.

Training information can be visualized using the browser by connecting to `localhost:6006`. 

## Test
Run inference on the test set using the Perceptual loss:
```
python test_nohtml.py --dataroot /data/datasets/scene --name scene --model pix2pixpl --dataset_mode aligned --load_iter 500 --force_test_output /data/results/scene_pl_500 --checkpoint_dir /data/checkpoints
```

Run inference on the test set using the default L1 loss:
```
python test_nohtml.py --dataroot /data/datasets/scene --name scene --model pix2pix --dataset_mode aligned --load_iter 500 --force_test_output /data/results/scene_500 --checkpoint_dir /data/checkpoints
```
Results will be saved in the `/data/results/scene_500` folder.

Change the `--load_iter` value to load weights from different epochs.


## Compute image quality metrics
First, compute the metrics using the L1 loss:
```
python src/metrics/compute_metrics.py /data/results/scene /data/results/scene
```
This will compute MSE, MI, MAE and SSIM metrics for each rendered/ground truth image pair and save results into a file named _metrics.pickle_.

Similarly, compute the metrics using the Perceptual loss:
```
python src/metrics/compute_metrics.py /data/results/scene_pl /data/results/scene_pl
```

Now compute the mean of all metrics:
```
python src/metrics/compute_mean_metrics.py /data/results/scene/metrics.pickle /data/results/scene_pl/metrics.pickle
```


