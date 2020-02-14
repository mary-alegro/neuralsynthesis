# Neural Render
This repository contains the code for the Neural Render project, developed during the AI Insight Data Science fellowship.
Neural Render is based on the [pix2pix](https://phillipi.github.io/pix2pix/) network architecture that has had great success performing tasks like style transfer and image
restoration. 

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

Run the app:
```
cd app
streamlit run neural_render.py
```
![neural render](./assets/app.gif)

## Train

## Test

## Compute metrics

