# LRCN
This repository contains files to load FP and insole data, build LRCN model, train and evaluate the model. Second chapter of my PhD dissertation.

# Code Description:
Desciption of the files:
- classes_Dataset.py: This file contains classes and functions required to load, clean and process data for different deep learning models. The output is an object conatining the data and all information required by the deep learning models to be built and trained. 
- classes_Models.py: This file contains classes and functions required to build different deep learning model objects.
- functions_preprocessing.py: Contains functions for general data processing. Mostly called by classes_Dataset.py
- functions_train.py: Contains files to train and tune the model. The model hyper-tuning is performed manually
- main_v04.py: The main file. The user-defined parameters are defined in this file. In creates the objects required to build, train and evaluate the model based on the inserted parameters. 
