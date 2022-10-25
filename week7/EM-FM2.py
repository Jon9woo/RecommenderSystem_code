# Created or modifed on Oct 2022
# Author: 임일
# Factorizagion Machines(FM) 연습문제 7-1-2 (EachMovie)

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')

# 사용자 정보를 읽고 column이름을 변경
users = pd.read_csv('C:/RecoSys/Data/person.csv', encoding='utf-8')
users.columns = (['user_id', 'age', 'sex', 'zip_code'])
users = users.drop(['zip_code'], axis=1)

# 영화 정보를 읽고 column이름을 변경
movies = pd.read_csv('C:/RecoSys/Data/movie.csv', encoding='utf-8')
movies = movies[['ID', 'Name', 'Action', 'Animation', 'Art_Foreign', 
                 'Classic', 'Comedy', 'Drama', 'Family', 'Horror', 'Romance', 'Thriller']]
movies.columns = (['movie_id', 'title',  'Action', 'Animation', 'Art_Foreign', 
                   'Classic', 'Comedy', 'Drama', 'Family', 'Horror', 'Romance', 'Thriller'])






K = 250
fm1 = FM(num_x, K, data, y, alpha=0.0007, beta=0.003, train_ratio=0.75, iterations=900, tolerance=0.0005, l2_reg=True, verbose=True)
result = fm1.test()
