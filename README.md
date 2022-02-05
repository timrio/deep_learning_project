# Sleep Stage Classification with Deep Learning

## Introduction

The aim of our project is to decode sign language with SIFT. Each letter of the alphabet corresponds to a sign : The American Sign Language letter with hand gestures includes 24 letters (excluding J and Z which require motion). We implement an algorithm taking in argument an image of a sign, and returning the corresponding letter. This project is very important, in order to help deaf and hard-of-hearing communicate in our society where being heard is the norm.

## Data

The data can be downloaded [here](https://www.kaggle.com/grassknoted/asl-alphabet), and must be unzip and moved to `.\Data\raw_data`

It consists of a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.

The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for  *SPACE* , *DELETE* and  *NOTHING* .

The test data set contains a mere 29 images.

## Structure

```

│   .gitignore
│   README.md
│   requirements.txt
│   run_models.ipynb
│   
├───Data
│   ├───pre_processed_data
│   │       Multitaper_eeg_test.npy
│   │       Multitaper_eeg_train.npy
│   │       Multitaper_position_train.npy
│   │       Multitaper_position_test.npy
│   │   
│   └───raw_data
│       │   sample_submission
│       │   X_test.h5
│       │   X_train.h5
│       │   y_train.csv
│       │   
│   
│   
└───src
    │   setup.py
    │   
    ├───sleep_classif
    │   │   CNNadvanced.py
    │   │   CNNmodel.py
    │   │   CNNmultitaper.py
    │   │   dataloaders.py
    │   │   LSTMConv.py
    │   │   preprocessing.py
    │   │   trainer.py
    

   
```

## Run code

Run the different models and data processings by running `run_models.ipynb`
