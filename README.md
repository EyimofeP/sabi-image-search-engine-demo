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