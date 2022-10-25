# Created or modified on Sep 2022
# author: 임일
# Demographic 기반 추천

import numpy as np
import pandas as pd

# Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')

# Load the u.items file into a dataframe
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 'unknown', 
          'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
          'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')

# movie ID와 title을 제외한 컬럼 지우기
movies = movies[['movie_id', 'title']]

# Load the u.data file into a dataframe
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1')

# timestamp 지우기
ratings = ratings.drop('timestamp', axis=1)

# Import the train_test_split function
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']

# Train/Test 데이터 나누기 (stratified 방식)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

# RMSE계산 함수
def RMSE(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# Baseline model (Best-seller 모델, 평균이 없을 경우에는 3)
def baseline(user_id, movie_id):
    try:
        rating = train_mean[movie_id]
    except:
        rating = 3.0
    return rating

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

train_mean = x_train.groupby(['movie_id'])['rating'].mean()
score(baseline)

# training set과 사용자 table을 결합(merge)
merged_data = pd.merge(x_train, users)
merged_data.head()

######Gender######

# gender별 평점 평균 계산
gender_mean = merged_data[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()

# user_id를 index로 설정
users = users.set_index('user_id')

# Gender기준 추천
def cf_gender(user_id, movie_id):
    # movie_id가 rating_matrix에 존재하는지 확인
    if movie_id in gender_mean:
        # gender 가져옴
        gender = users.loc[user_id]['sex']
        # 해당 영화에 해당 gender의 평균값이 존재하는지 확인
        if gender in gender_mean[movie_id]:
            # 해당 영화의 해당 gender의 평균값을 예측값으로 함
            gender_rating = gender_mean[movie_id][gender]
        else:
            gender_rating = 3.0
    else: # movie_id가 rating_matrix에 없으면 기본값 3.0을 예측값으로 함
        gender_rating = 3.0
    return gender_rating

score(cf_gender)

######Occupation######

occupation_mean = merged_data[['movie_id', 'occupation', 'rating']].groupby(['movie_id', 'occupation'])['rating'].mean()

def cf_occupation(user_id, movie_id):
    if movie_id in occupation_mean:
        occupation = users.loc[user_id]['occupation']
        if occupation in occupation_mean[movie_id]:
            occupation_rating = occupation_mean[movie_id][occupation]
        else:
            occupation_rating = 3.0
    else:
        occupation_rating = 3.0
    return occupation_rating

score(cf_occupation)






