# Deep learning algorithm predicts diabetic retinopathy progression in individual patients 

<br>

## Description
This repository collect all the scripts used to re-create the field-specific convolutional neural networks (CNN) pillars presented in the manuscript: *F. Arcadu, F. Benmansour, A. Maunz, J. Willis, Z. Haskova, and M. Prunotto, "Deep learning algorithm predicts diabetic retinopathy progression in individual patients", Nature Digital Medicine, 2019*

<br>

## Requirements

Python-2.7 , Tensorflow-1.4.0. , Keras-2.2.4 , pandas-0.24.2 , skimage-0.14.3 , sklearn-0.20.3 , progressbar-2.5

<br>

## Folders
* __create_cnn_pillars__ contains the scripts to train with transfer-learning followed by fine-tuning the 7 CNN field-specific pillars, 1 for each color fundus field-of-view (FOV);
* __cnn_configs__ contains the configs files used to the train the CNN related to a specific FOV and a specific month of DR progression;

<br>

## Training Separate CNN Pillars
Command line to train a single CNN model:
```
python cnn_train.py \ 
  -i1 < training CSV > 
  -i2 < testing CSV >
  -p < YAML comfig file >
  -o < output path > \ 
  -col-imgs < column containing the image filepaths > \
  -col-label < column containing the image labels >
```

The training will produce 3 output files: 
* a JSON file with the training history;
* a YAML file that is a copy of the config file used for training; 
* a HDF5 which contains the best model saved according to the chosen metric;
* a CSV logging all metrics related to the training; it is updated online.

For more information, including additional runnable examples, type 
```
python cnn_train.py -h
```
To complete a quick run to test whether everything works fine add the flag
```
--debug
```

<br>

## Compute forward predictions using trained CNN
The routine `cnn_predict.py` allows to run the forward prediction provided a single image in input. Command line:
```
python cnn_predict.py \
   -i < YAML file for cnn_run_predict >
   -o < select path where to save file >
   -m < select trained HDF5 model >
   -t < select type of architecture >
```

<br>

### Author
* **Filippo Arcadu** - July 2019

<br>

### Last Update
29.07.2019
