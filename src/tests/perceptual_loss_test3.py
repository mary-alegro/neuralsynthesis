from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from src.models.perceptual_loss import PerceptualLoss
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    #optimizer = optim.LBFGS([input_img.requires_grad_()])
    LR = 1e-2
    optimizer = optim.Adam([input_img.requires_grad_()],LR)
    return optimizer


if __name__ == '__main__':
    style_img = image_loader("./picasso.jpg")
    #content_img = image_loader("./dancing.jpg")
    content_img = style_img.clone()

    input_img = image_loader("./dancing.jpg")
    # if you want to use white noise instead uncomment the below line:
    #input_img = torch.randn(content_img.data.size(), device=device)
    # add the original input image to the figure:
    plt.figure()
    imshow(input_img, title='Input Image')

    ploss = PerceptualLoss().to(device)

    num_steps = 300
    step = 0

    optimizer = get_input_optimizer(input_img)

    while step <= num_steps:
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()

        style_score,content_score = ploss(input_img, content_img, style_img)
        loss = style_score + content_score
        loss.backward()

        step += 1

        print("run {}:".format(step))
        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            style_score.item(), content_score.item()))
        print()

        optimizer.step()
        input_img.data.clamp_(0, 1)



    plt.figure()
    imshow(input_img, title='Result Image')

    plt.ioff()
    plt.show()