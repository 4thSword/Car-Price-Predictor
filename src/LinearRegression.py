#Imports
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Gloal constants:
TRAIN_PATH = "../input/processed_train.csv"
TEST_PATH = "../input/processed_test.csv"

# Functions definition:


# Execution:
if __name__ == "__main__":
    #Data preparing
    data_train = training.drop(['Id','price'],axis = 1)
    data_test = test.drop(['Id'],axis = 1)    
    y_train = training['price']

    #Model Training:
    
    l_reg = LinearRegression()
    l_reg.fit(data_train, y_train)

    #Applying trained model to our train set:
    y_train_pred = mlp.predict(data_test)

    #Result tratement to be submmited:
    submission = pd.DataFrame({
        'Id':test['Id'],
        'Price': y_train_pred
    })

    # Generating output file:
    submission.to_csv('output/submission.csv',index=False)