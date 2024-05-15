import streamlit as st # library for deploying ml app

# import tensorflow as tf
import tf_keras as tf # deep learning framework
import tensorflow_hub as hub #used to load saved model
import numpy as np # for numerical computation

# for image mainpulation
import cv2 
from PIL import Image

# setting title bar and page layout
st.set_page_config(layout="wide", page_title="Sabi Image Classifier Demo")

# Frontend Texts
st.title("Sabi Product Finder")
st.subheader("You no *Sabi* the name of your Product? Snap Am, We go *Sabi* am for you")
st.text("Send a picture to me and I'll predict")

# loading saved model
# model = tf.keras.models.load_model("vanilla.keras")
model = tf.models.load_model(
       ("sa.h5"),
    #    ("sabi_image_classifier.h5"),
       custom_objects={'KerasLayer': hub.KerasLayer}
)

# labels that can be predicted
labels = ["Peak Milk Powder 400g", "Nestle Milo Chocolate 1.8kg", "Golden Penny Sugar 500g", "Not a Product",]


def predict(picture):    
    image = cv2.resize(np.array(picture), (224, 224)) # resize image to tensor of 224 * 224 pixels
    image = image / 255 # scale the image

    try:
        x = np.reshape(image, (-1, 224, 224, 3)) #if tensor is RGB (,3), process for prediction
    except:
        x = np.reshape(image, (1, 224, 224, 1)) #if tensor is not RGB, convert then process for prediction
        x = np.concatenate([x]*3, axis=-1)

    x = model.predict(x).flatten() # predict value and get predictions
    prediction = x.argmax() # Get the label of the highest accuracy 
    confidences = {labels[i]: float(x[i]) for i in range(4)} # get prediction for each label and store in a dictionary
    # if confid
    return prediction, confidences


# predict (picture)
upload= st.file_uploader('Insert image for classification', type=['png','jpg', 'jpeg']) # for user to upload picture
c1, c2= st.columns(2, gap="large") # creating 2 columns


if upload is not None:
    img = Image.open(upload) # collect image
    
    c1.header('Your Picture')
    c1.image(img) # show image on screen
    answer, pred = predict(img) # predict label image

    c2.header('Predictions :')
    c2.subheader(f"{labels[answer]}") # display  prediction
    
    # get the 3 most accurate predictions
    sorted_values = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:1]

    # Loop over the sorted values and create a progress bar for each label and prediction
    for key, value in sorted_values:
        c2.write(key)
        progress_bar = c2.progress(0)
        for i in range(1):
            progress_bar.progress(value, text=f"{value * 100:.2f}% confidence ")
