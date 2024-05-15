import streamlit as st # library for deploying ml app

import tf_keras as tf # deep learning framework (using oldee=r version because of pretrained model)
import tensorflow_hub as hub #used to load saved model
import numpy as np # for numerical computation

# for image mainpulation/conversion/normalization
import cv2 
from PIL import Image

# setting title bar and page layout of the web page
st.set_page_config(layout="wide", page_title="Sabi Image Classifier Demo")

# Frontend Texts for web page
st.title("Sabi Product Finder")
st.subheader("You no *Sabi* the name of your Product? Snap Am, We go *Sabi* am for you")
st.text("Send a picture to me and I'll predict")

# loading saved model
model = tf.models.load_model(
       ("sabi_final_model.h5"),
       custom_objects={'KerasLayer': hub.KerasLayer}
)

# Products labels that can be predicted
labels = ["Peak Milk Powder 400g", "Nestle Milo Chocolate 1.8kg", "Golden Penny Sugar 500g", "Not a Product",]


# Function used to predict the product of a picture
def predict(picture):    
    # first convert the image to a tensor then resize image to tensor of 224 * 224 pixels
    image = cv2.resize(np.array(picture), (224, 224)) 

    # normalize the picture for image scaling
    image = image / 255 

    try:
        # if tensor is RGB format i.e has 3 dimensions, process for prediction
        x = np.reshape(image, (-1, 224, 224, 3)) 
    except:
        # else if tensor is not RGB, convert to RGB then process for prediction
        x = np.reshape(image, (1, 224, 224, 1)) 
        x = np.concatenate([x]*3, axis=-1)

    # predict the image and get predictions of the product
    x = model.predict(x).flatten() 

    # Get the product of the highest accuracy 
    prediction = x.argmax() 

    # Get the confidence level of the product label
    confidence = x[0] 
  
    """ 
    Here we are setting a threshold, 
    if the accuracy of the product is above 90%, Show the product to the user 
    Else if the accuracy is less than 90%, Label as Not a Product
    """
    if confidence > 0.9:
        return prediction, confidence
    else:
        return 3, confidence


# This is to create a button/file I/O where the user inserts their picture in either png/jpg/jpeg formats
upload= st.file_uploader('Insert image for classification', type=['png','jpg', 'jpeg'])

"""
Dividing the page into two sections, 
one to show the user's image at the left division
and the preiction at the right division
"""
c1, c2= st.columns(2, gap="large") 


# Checking if the user didnt upload a corrupt/bad picture
if upload is not None:
    # Collect the uploaded image
    img = Image.open(upload)
    
    c1.header('Your Picture')

    # show image on left side of the screen
    c1.image(img) 

    # Predict the image and get the product label and confidence
    answer, pred = predict(img)

    c2.header('Predictions :')

    # Display prediction at the right side of the screen
    c2.subheader(f"{labels[answer]}") 
    

    # If prediction of the product is above 90%, display a graphic with accuracy 
    if pred > 0.9:
        progress_bar = c2.progress(0)
        progress_bar.progress(float(pred), text=f"{float(pred) * 100:.2f}% confidence")
    

    # Allow user input for product price and minimum order quantity
    price = st.number_input("Product Price:", min_value=0.0, step=0.01, format="%.2f")
    min_order_quantity = st.number_input("Minimum Order Quantity:", min_value=1, step=1, format="%d")
    
    # Process the user inputs here to meet our needs

