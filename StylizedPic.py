import functools
import os
import io

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import PIL
import base64


# Load images from user

uploaded_content_pic = st.sidebar.file_uploader("Upload a content picture to Artify ")
uploaded_style_pic = st.sidebar.file_uploader("Upload your own artistic style picture")
choose_default_style = st.sidebar.selectbox(
    "Or choose a default artistic style", (
        'Nothing selected', 
        'Merello - Woman with a cat', 
        'Van Gogh - Starry night',
        'Munch - Scream',
        )
    )
output_image_size = int(st.sidebar.select_slider(
        "Select pixel size of stylized picture",
        options = [350, 600, 900, 1200],
        value= 600,
    )
)

if uploaded_content_pic is not None and (uploaded_style_pic is not None or choose_default_style != 'Nothing selected'): 
    uploaded_content_pic = uploaded_content_pic.getvalue()
    
    if uploaded_style_pic is not None:
        uploaded_style_pic = uploaded_style_pic.getvalue()
    else:
        if choose_default_style == 'Merello - Woman with a cat':
                file = 'merello_woman.jpg'
                image = open(file, 'rb')
                image_read = image.read()
                uploaded_style_pic = image_read
        elif choose_default_style == 'Van Gogh - Starry night':
                file = 'starry_night.jpg'
                image = open(file, 'rb')
                image_read = image.read()
                uploaded_style_pic = image_read
        elif choose_default_style == 'Munch - Scream':
                file = 'scream.jpg'
                image = open(file, 'rb')
                image_read = image.read()
                uploaded_style_pic = image_read
    
    # Define output image size - the higher the pixel value the more style patterns 
    # will be repeated in the Stylized picture
    
    content_img_size = (output_image_size, output_image_size)
    style_image_size = 256
    style_img_size = (style_image_size, style_image_size)

    # Function to cropimage to feed it into the model (hub_module)
    def crop_center(image):
        """Returns a cropped square image."""
        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2
        offset_x = max(shape[2] - shape[1], 0) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, new_shape, new_shape)
        return image

    # Function to load image and preprocess it
    @functools.lru_cache(maxsize=None)
    def load_image_local(image, image_size=(256, 256), preserve_aspect_ratio=True):
        """Loads and preprocesses images."""
        # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
        img = tf.io.decode_image(image, channels=3, dtype=tf.float32)[tf.newaxis, ...]
        img = crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img

    # Load TF Hub model which will blend the content and art pictures into a desired stylized picture 
    @st.cache(suppress_st_warning=True)
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)


    content_image = load_image_local(uploaded_content_pic, content_img_size) 
    style_image = load_image_local(uploaded_style_pic, style_img_size)

    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # Functions which converts stylized image to PIL image format
    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    final_img = tensor_to_image(stylized_image)

    st.header("Your Stylized picture")
    st.markdown("You can change the pictures or the pixel size and a new picture will be produced."
                " To use the default selection, you must delete any style pictures uploaded by clicking (x)")
      
    def get_image_download_link(img):
        """Generates a link allowing the PIL image to be downloaded
        in:  PIL image
        out: href string
        """
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/jpg;base64,{img_str}" download="image.jpg">Download stylized picture</a>'
        return href

    st.markdown(get_image_download_link(final_img), unsafe_allow_html=True)
    st.image(final_img, use_column_width=True)
else:
    st.image('Welcome.png', use_column_width=False)
