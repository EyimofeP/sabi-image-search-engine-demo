# Sabi Image Classification Search Engine - Demo
___

The aim of this project is to create an application where wholesalers can identify the product variants of the app they want to upload to the Sabi Marketplace

It currently identifies this 3 products variants:
1. Peak Milk Powder 400g
2. Nestle Milo Chocolate 1.8kg
3. Golden Penny Sugar 500g

Other products that do not fall into this category are identified as "Not Listed Products"

___
### Machine Learning Models

There are  3 models
1. vanilla.keras - Our Base CNN model with no pretrained model
2. sabi_image_classifier.h5 - A pretrained MobileNetV2 model with no finetuning
3. sabi_final_model.h5 - A pretrained MobileNetV2 model with fine tuning to reduce overfitting that occured in the past 2 models, this was the final model used for predictions


##SABI IMAGE CLASSIFIER

This model was trained on an existing pretrainer model. here is the link: https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow2/130-224-classification/

About Pretrainer Model: This part of the model consists of a MobileNet V2 base model loaded from TensorFlow Hub. It takes input images of size (224, 224, 3) and freezes the weights (trainable=False). The MobileNet V2 base model serves as a feature extractor.

Additional Layers: After the MobileNet V2 base model, there are additional layers added for fine-tuning and classification. These layers include a dropout layer (tf.layers.Dropout(0.2)), a flatten layer (tf.layers.Flatten()), a dense layer with 64 units (tf.layers.Dense(64)), and a final dense layer with 4 units for classification with softmax activation (tf.layers.Dense(4, activation="softmax")).

Compilation and Training: The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. It is then trained on the training data (X_train, y_train) for 10 epochs.

Evaluation: After training, the model is evaluated on the test data (X_test, y_test), and predictions are generated using the predict method.

Prediction Function: The predict function takes an image as input, preprocesses it, and generates predictions using the trained model. It returns the predicted class label along with the confidence score.

Saving the Model: At the end of the script, the trained model is saved to a file named "sabi_image_classifier.h5".
