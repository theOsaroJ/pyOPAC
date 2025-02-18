#!/bin/bash


#In no particular order but use based on intuition which preprocessing data --> training model (make sure to put targets.csv in format as in example which is the y of the training molecules in ~./data) --
# --> computing descriptors for new molecules after running python modify.py and moving file to right place ( see creating_xyz_files) for more details --
# --> predicting properties of new molecule.
## --------------------------------- first make sure the multi-mol xyz of training data (xyz) is done in right form ---------------------###
cd creating_the_xyz/

# assuming you have a train_example.xyz --> see the format of our multimolecule xyz file

python3 modify.py train_example.xyz train.xyz    

# create dataset/training_xyz_files/ in root directory or wherever
cp train.xyz ../dataset/training_xyz_files/train.xyz
cd ..

### --------------------------------- preprocessing training xyz into descriptors gotten from RDKit ------------------------------------- ###
python3 opac/scripts/preprocess_data.py \
    --input-dir dataset/training_xyz_files/ \
    --targets-file dataset/targets.csv \
    --output-descriptors dataset/descriptors.csv

### ---------------------------------- train a vae model ------------------------------------ ###
python3 opac/scripts/train_model.py \
    --descriptors-file dataset/descriptors.csv \
    --targets-file dataset/targets.csv \
    --model-output saved_models/trained_model.pth \  #create a saved_models dir
    --epochs 200 \
    --test-size 0.2 \
    --learning-rate 0.001 \
    --batch-size 64 \
    --hidden-dim 512 \
    --weight-decay 1e-4

## ------------------------------------- get new descriptors of test molecules in right xyz format (same for training) ---------------------------##
cd creating_the_xyz/
python modify.py test_example.xyz test.xyz
cp test.xyz ../dataset/testing_xyz_files/test.xyz
cd ..

## --------------------------------------- convert the test xyz to the descriptors recognized by model------------------------##
python opac/scripts/compute_descriptors.py \
    --input-dir dataset/testing_xyz_files/ \
    --output-descriptors dataset/new_descriptors.csv

## ------------------------------------------ make predictions of the new molecules ----------------------------------------- ##
python opac/scripts/predict_properties.py \
    --model-file saved_models/trained_model.pth \
    --descriptors-file dataset/new_descriptors.csv \
    --predictions-output dataset/predictions.csv

## If you want to do Active Learning ##

## ---------------------------- running active learning ---------------------------- ##
python opac/active_learning/active_learning.py \
    --descriptors-file dataset/descriptors.csv  \
    --targets-file dataset/targets.csv  \
    --initial-train-size 1000  \
    --query-size 5   --iterations 2  \
    --model-output saved_models/al_trained_model.pth \
    --hidden-dim 128   \
    --epochs 50   \
    --batch-size 32  \
    --learning-rate 1e-3  \
    --weight-decay 1e-4

## ---------------------------predict the properties of new molecules with the AL model -----------------------##
python opac/active_learning/predict_new_data.py \
    --model-file saved_models/al_trained_model.pth \
    --descriptors-file dataset/new_descriptors.csv \
    --predictions-output dataset/new_predictions.csv
