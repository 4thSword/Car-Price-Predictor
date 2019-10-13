# Car-Price-Predictor

## Overview

This project is based on a Kaggle competition proposed in data analytics class where the goal is to predict the price of a car based on their features and using a provided [dataset](https://www.kaggle.com/c/datamad0819-vehicles/data) and applying differents agorithms of machine learning find the result which fits with a hidden answer stored in Kaggle.

## The Project

In this project two datsets are provided: One with the answer that will be used to train our model, and another without answer that will be used to apply our model trained and make a prediction to be submitted to kaggle.

### Structure and Files

This project is structured in three differents folders:

* __input.-__ In this folder we will allocate both datasets(cars_train.csv and cars_test.csv), and after a first data processing process two new files will be generated and allocated there, with our data clean and ready to be used to fit an predict with the differents models.


* __output.-__ In this folders two files will be allocated. A first file with a log of the results of the differents cleaning process in different ways and the result of the RMSE (root mean squared error) provided by the differents models used and their modifications, and will be used as a guide to select the best model with the best hyper-paramenters configuration. A second file called Submission.csv with the prediction to be submitted to Kaggle.

* __src.-__ In this folder the differents scripts will be allocated.

