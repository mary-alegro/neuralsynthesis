import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
from PIL import Image
import skimage.io as io
import run_inference as ri
import urllib
import sys

#print(sys.path)


def save_image(image):
    image_np = np.array(image)
    image_np = image_np[..., :3]
    if image_np.shape[1] < 512: #not in pix2pix AB format
        image_np = np.concatenate((image_np, image_np), axis=1)
        A = image_np
        B = None
    else:
        A = image_np[:,0:256,:]
        B = image_np[:,256:512,:]
    io.imsave('./db/test/tmp.png', image_np)
    return A,B


def get_file_content_as_string(path):
    url = 'file://' + os.path.abspath(path)
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def main():

    readme_text = st.markdown(get_file_content_as_string("about.md"))
    data = st.sidebar.file_uploader('Upload a file')
    # app_seg = st.sidebar.selectbox("Use segmentation?",
    #                                 ["No", "Yes"])
    app_model = st.sidebar.selectbox("Choose model",
                                    ["Pix2Pix L1 norm", "Pix2Pix Perceptual Loss"])
    app_epoch = st.sidebar.selectbox("Select epoch",
                                    ['500','10','100', '200', '300','400'])


    if data:
        readme_text.empty()

        #download image locally and save it in the appropriate format
        image = Image.open(data)
        A,B = save_image(image)
        if B is None:
            st.image([A], caption=['Rendered image'])
        else:
            st.image([A,B], caption=['Rendered image','Ground truth image'])

        # #use segmentation?
        # if app_seg == 'Yes':
        #     use_seg = True
        # elif app_seg == 'No':
        #     use_seg = False
        # else:
        #     st.sidebar.success('You must select one model.')

        #get model
        if app_model == 'Pix2Pix L1 norm':
            model = 'pix2pix'
        elif app_model == 'Pix2Pix Perceptual Loss':
            model = 'pix2pixpl'
        else:
            st.sidebar.success('You must select one model.')

        #get epoch
        if app_epoch == '10':
            epoch = 10
        elif app_epoch == '100':
            epoch = 100
        elif app_epoch == '200':
            epoch = 200
        elif app_epoch == '300':
            epoch = 300
        elif app_epoch == '400':
            epoch = 400
        elif app_epoch == '500':
            epoch = 500

        #run prediction
        new_image = ri.predict('./checkpoints','./db',model,epoch)

        # #apply mask
        # if use_seg:
        #     new_image = apply_mask(new_image)

        #st.image([new_image], caption=['Reconstructed'])
        st.image([new_image], caption=['Reconstructed image'])

    # else:
    #     # Explain your project nicely
    #     """
    #     Neural Render uses Generative Adversarial Networks to repair poorly rendered 3d scenes.
    #     It aims to works as a post-processing step, enhancing your renderings and avoiding the need to re-run your 3d pipelines.
    #
    #     👈 **Please select an image and model on the left and let the AI magic will happen.** :)
    #     """

#create GIF from demo
#ffmpeg -ss 00:00:00.000 -i demo_app.mov -pix_fmt rgb24 -r 10 -s 2560x1600 -t 00:00:35.000 output.gif

if __name__ == '__main__':
    main()