# Imports:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Global Constants:

TRAIN_PATH = "../input/cars_train.csv"
TEST_PATH = "../input/cars_test.csv"
OUTPUT_TRAIN_PATH = "../input/processed_train.csv"
OUTPUT_TEST_PATH = "../input/processed_test.csv"

# Functions definitions:

def load_data(path):
    return pd.read_csv(path)

def clean_dataframe(df):
    # Clean column by column all data to be able to let our model work properly.
    
    # Dropping columns that I think that could't give us enogh infomration for our prediction
    df.drop(['county_fips','state_fips','state_code','state_name','city','weather','make','county_name','manufacturer'],axis=1, inplace=True)
    # Cleaning the cylinders column
    df['cylinders'] = df.cylinders.str.strip('cylinders')
    df['cylinders'] = df.cylinders.str.replace('oth', '0')
    df['cylinders'] = df['cylinders'].fillna('0')
    df['cylinders'] = pd.to_numeric(df['cylinders'])
 
    # Cleaning the odometer column filling empty data with de median to try to avoid let the outliers get importance in our predcition
    df['odometer'] = df.odometer.fillna(df.odometer.median())
    # Filling nulls in categorical variables by unknown:
    categorical = ['drive','size','paint_color','type']
    for category in categorical:
        df[category] = df[category].fillna('unknown')    
    # filling the empty values of year category with the median in order to don't let our model to be affected by the outliers.
    df['year'] = df['year'].fillna(training['year'].median())
    # getting dummies to be able to apply our model
    df = pd.get_dummies(df)
    for column in df:
        if "unknown" in column:
            df.drop(columns = [column],inplace=True)
    return df



if __name__ == "__main__":
    
    #First step: To load our training data and our test data:
    
    training = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)
    

    #Second step: To clean data before applying standarization:

    training_clean = clean_dataframe(training)
    test_clean = clean_dataframe(test)
    

    #Third step: Standarization

    standarize = StandardScaler()
    standarize.fit(training_clean[['year','odometer','cylinders','long','lat']])
    training_clean[['year','odometer','cylinders','long','lat']]= standarize.transform(training_clean[['year','odometer','cylinders','long','lat']])
    test_clean[['year','odometer','cylinders','long','lat']]= standarize.transform(test_clean[['year','odometer','cylinders','long','lat']])
 

    # Frouth step: Export data to csv

    training_clean.to_csv(OUTPUT_TRAIN_PATH,index=False)
    test_clean.to_csv(OUTPUT_TEST_PATH,index=False)
    

    # Adding metrics to a log, for next study of better model.
    
    with open('../output/log.txt',"a+") as f: 
        f.write("DATA PROCESSING | COLUMNS BEFORE PCA: {} \n".format(training.columns)) 