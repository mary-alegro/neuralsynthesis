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

## Train

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


