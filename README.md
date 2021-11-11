# StylizedPic

This is an app which lets you take any picture and modify it in the style of a painting or art picture. 

The app is hosted on Stremlit https://share.streamlit.io/alexadamov/stylizedpic/main/OwnArt.py

## Running locally
If you want to use the app locally and modify it, install Streamlit
https://docs.streamlit.io/library/get-started/installation#install-streamlit-on-windows 

then run in local code editor

The app should pop in your browser on a local port and then you can modify it as you wish. 

```
streamlit run OwnArt.py
```

## Machine learning component

The app uses a pretrained Style transfer network on Tensorflow. It picks up the model from https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2

This is an implementation based on the following research paper. You should check itout as it an ease read actually. 

Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens. Exploring the structure of a real-time, arbitrary neural artistic stylization network. Proceedings of the British Machine Vision Conference (BMVC), 2017.
