# Created or modified on Sep 2022
# author: 임일
# Demographic 기반 추천 실습과제 w3-1

import numpy as np
import pandas as pd

# Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')
users.head()

# Load the u.data file into a dataframe
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings.head()

# Drop the timestamp column
ratings = ratings.drop('timestamp', axis=1)

# Import the train_test_split function
from sklearn.model_selection import train_test_split
# Assign x as the original ratings dataframe and y as the user_id column of ratings
x = ratings.copy()
y = ratings['user_id']
# Split into training and test datasets, stratified along user_id
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

# RMSE계산 함수
def RMSE(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# 주어진 추천 알고리즘(model)의 RMSE를 계산하는 함수
def score(model):
    # Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])
    # Predict the rating for every user-movie tuple
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    # Extract the actual ratings given by the users in the test data
    y_true = np.array(x_test['rating'])
    # Return the final RMSE score
    return RMSE(y_true, y_pred)

# Merge the original users dataframe with the training set
merged_data = pd.merge(x_train, users)

# user_id를 index로 설정
users = users.set_index('user_id')

######Gender&Occupation######

# Compute the mean rating of every movie by gender and occupation
gen_occ_mean = 

# Gender and Occupation Based Collaborative Filter using Mean Ratings
def cf_gen_occ(user_id, movie_id):
    return

score(cf_gen_occ)
