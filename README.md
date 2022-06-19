# Image_Captioning

Image captioning can be regarded as an end-to-end Sequence to Sequence problem, as it converts images, which is regarded as a sequence of pixels to a sequence of words. For this purpose, we need to process both the language or statements and the images.

Here two models one sequence model and other convolution model is used.
Sequence model is lstm, convolution model is Xception net.

This model has been trained on Flicker8k dataset. Link to zip file of dataset is provided here :

Link to dataset folder : https://drive.google.com/file/d/1JuV8VI1fcGE7fSsxjimFMm4Z4HS7Ggoi/view?usp=sharing

Link to text,descriptions,train_imgs name, test imgs name: https://drive.google.com/file/d/1DjN6bVURuGX4tABM8sxARhof3yxrq3Jk/view?usp=sharing

Unzip the text zip file in a folder.

You can see the output results in image_caption_eg_result

model weights and other required files are stored in model_res folder
For Sample In input folder one image is present.

Two types of infer scripts are provided:
1. infer.py, which can be used to perform infer on the test files from flicker.txt.zip file.

To run infer:

!python path_to_infer_script [PATH TO FLICKER TEXT FOLDER] [PATH TO FLICKER DATASET FOLDER]
(infer-script : infer.py)

2. custom_infer.py, which can be used to test on any image that you want to.

To run custom infer:

!python path_to_custom_infer_script [PATH TO IMAGE DATA FOLDER]
(custom_infer: custom_infer.py)

Demo for both types of infer is provided in demo_infer_imgcap.py

To train the model:

!python path_to_training_script(train.py) [PATH_TO_FLICKER_TEXT_FOLDER] [PATH_TO_FLICKER_IMAGE_FOLDER] [NUMBER_OF_EPOCHS_TO_RUN]

(training-script : train.py)
Demo for training is provided in demo_train_imgcap.py
