# Package Imports
import time #This is for timing the code
StartTime = time.time()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #eliminates future version warnings 
import pandas as pd
import seaborn as sns #These are the packages for the random forest, we must write the code for the neural network. 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression #This is the random forest classifier
from sklearn.datasets import make_classification #This is a package that generates random data for testing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from pickle import dump
import numpy as np
import random as rn
import os
from sklearn.preprocessing import LabelEncoder  #This is for encoding the output data
pd.options.mode.chained_assignment = None  # default='warn'
current_dir = os.path.dirname(os.path.realpath(__file__))

# ******************** Import Data ***************************************
Training_df = pd.read_excel("output.xlsx")
results = pd.DataFrame() #This is the dataframe that will store the results of the model training. This is added to solve a concatenation issue of the parameters
rfscores = {}
scores = {}
r2Train = {}
# ******************** Data Organization ***************************************
X = np.array(Training_df[['Fx1' , 'Fy1', 'Fz1', 'Tx1', 'Ty1', 'Tz1', 'Fx2', 'Fy2', 'Fz2', 'Tx2', 'Ty2', 'Tz2', 'Fx3', 'Fy3', 'Fz3', 'Tx3', 'Ty3', 'Tz3', 'Fx4', 'Fy4', 'Fz4', 'Tx4', 'Ty4', 'Tz4', 'Fx5', 'Fy5', 'Fz5', 'Tx5', 'Ty5', 'Tz5', 'Fx6', 'Fy6', 'Fz6', 'Tx6', 'Ty6', 'Tz6', 'Fx7', 'Fy7', 'Fz7', 'Tx7', 'Ty7', 'Tz7', 'Fx8', 'Fy8', 'Fz8', 'Tx8', 'Ty8', 'Tz8', 'Fx9', 'Fy9', 'Fz9', 'Tx9', 'Ty9', 'Tz9', 'Fx10', 'Fy10', 'Fz10', 'Tx10', 'Ty10', 'Tz10', 'Fx11', 'Fy11', 'Fz11', 'Tx11', 'Ty11', 'Tz11', 'Fx12', 'Fy12', 'Fz12', 'Tx12', 'Ty12', 'Tz12', 'Fx13', 'Fy13', 'Fz13', 'Tx13', 'Ty13', 'Tz13', 'Fx14', 'Fy14', 'Fz14', 'Tx14', 'Ty14', 'Tz14', 'Fx15', 'Fy15', 'Fz15', 'Tx15', 'Ty15', 'Tz15']])
Y = np.array(Training_df[['Collision Type']]) #This is the output data that the model will predict.
LabelEncoder = LabelEncoder()
Y = LabelEncoder.fit_transform(Y)

# ******************** Random Forest Model ***************************************
for seed in range(10):    #trains 10 models. The seed determines how initilization and randomization of each model is done

    os.environ['PYTHONHASHSEED'] = '0'   #To rn.seed, this code keeps randomization and splitting concistent for each model run
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    np.random.seed(seed)
    rn.seed(seed)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed) 
    #test size can be changed but not to beyond .30. This symbolizes %  of dataset used for testing, and the rest is training. 

    #Merge X_train, X_test, y_train, Y_test and save per iteration
    seeded_df = [X_train, Y_train, Y_train, Y_test]
    scalerX = StandardScaler() #This is data normalization for data that changes larger orders of magnitude.
    scalerX.fit(X) #This fits the scaling to the X-Data  
    X_train = scalerX.transform(X_train) 
    X_test = scalerX.transform(X_test)
    Y_train = Y_train.ravel()

    lr = LogisticRegression()
    Score = make_scorer(accuracy_score)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
              'penalty': ['l1', 'l2'],
              'dual': [True, False]}

    lr_grid = GridSearchCV(estimator=lr, param_grid=param_grid, scoring=Score, n_jobs=-1, return_train_score=True)
    lr_grid.fit(X_train, Y_train.reshape(-1,))
    # Determine Model prediction for train and test
    cv_results = lr_grid.cv_results_

    ScoreTrain = lr_grid.score(X_train, Y_train) 
    #rSquaredtrain = print('nn Train Score of' + str( seed ) + ' is ' + str(ScoreTrain))
    ScoreTest = lr_grid.score(X_test, Y_test)
    #rSquaredtest = print('nn Test Score of' + str( seed ) + ' is ' + str(ScoreTest))
    scores[seed] = (ScoreTrain, ScoreTest)
    modelNumber = seed
    predictionTrain = lr_grid.predict(X_train)
    R2score = r2_score(Y_train, predictionTrain)

   # Get the model's parameters
    params = lr_grid.get_params()
    

    # Get the model's parameters
    params['Train Score'] = ScoreTrain
    params['Test Score'] = ScoreTest
    params['R2 Score'] = R2score
    params['Model Number'] = modelNumber
    
    # Append the results to the DataFrame
    params_df = pd.DataFrame([params])
    results = pd.concat([results, params_df], ignore_index=True)
    results.to_csv('lr_results.csv', index=False)
    SeededDictRF = {'model': modelNumber, 
                    'X_train': X_train, 
                    'X_test': X_test, 
                    'Y_train': Y_train, 
                    'Y_test': Y_test,
                    'Train Score': ScoreTrain,
                    'Test Score': ScoreTest}
    
    pickle_file_path = os.path.join(current_dir, 'lr_model_XY_TrainTest' + str(seed) + '.pkl')
    pickle_model_path = os.path.join(current_dir, 'lr_model' + str(seed) + '.pkl')
    dump(SeededDictRF, open(pickle_file_path, 'wb'))
    # ******************** Save Model ***************************************
    dump(lr_grid, open(pickle_model_path, 'wb')) #The pickle load function will be able to pull the model     

# ******************** Time the Model ***************************************
EndTime = time.time()
RunTime = EndTime - StartTime
print(f"The runtime of the program is {RunTime} seconds.")




