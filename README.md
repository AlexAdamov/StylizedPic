# StylizedPic

This is an app which lets you take any picture and modify it in the style of a painting or art picture. 

The app is hosted on Streamlit https://share.streamlit.io/alexadamov/stylizedpic/main/StylizedPic.py

## Running locally
If you want to use the app locally and modify it, first install Streamlit (Python library)
https://docs.streamlit.io/library/get-started/installation#install-streamlit-on-windows 

Then type in cmd / Terminal the below command. The app should pop in your browser on a local port and then you can modify it as you wish. 

```
streamlit run OwnArt.py
```

## Machine learning component

The app uses a pretrained Style Transfer Network implemented in Tensorflow. It picks up the model from https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2

This is an implementation based on the following research paper. I recommend to check it out as it an easy and insightful read. 

Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens. Exploring the structure of a real-time, arbitrary neural artistic stylization network. Proceedings of the British Machine Vision Conference (BMVC), 2017. https://arxiv.org/abs/1705.06830
