#Imports
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gloal constants:
TRAIN_PATH = "../input/processed_train.csv"
TEST_PATH = "../input/processed_test.csv"

# Functions definition:


# Execution:
if __name__ == "__main__":
    #Data preparing
    data_train = training.drop(['Id','price'],axis = 1)
    data_submit = test.drop(['Id'],axis = 1)    
    y = training['price']

    X_train, X_test, y_train, y_test = train_test_split(data_train,y, test_size=0.1)

    #Model Training:
    
    l_reg = LinearRegression()
    l_reg.fit(X_train, y_train)

    #Applying trained model to our train set:
    y_test_pred = l_reg.predict(X_test)
    #checking the error
    print(mean_squared_error(y_test,y_train_pred))

    # Applying model to our submission set:
    y_pred = l_reg.predict(data_submit)



    #Result tratement to be submmited:
    submission = pd.DataFrame({
        'Id':test['Id'],
        'Price': y_pred
    })


    
    # Generating output file:
    submission.to_csv('output/submission.csv',index=False)