# Created or modified on Sep 2022
# author: 임일
# Simple CF with EachMovie

import numpy as np
import pandas as pd

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')




score(cf_simple)



###################### 추천하기 ######################
# 추천을 위한 데이터 읽기 (추천을 위해서는 전체 데이터를 읽어야 함)

movies = pd.read_csv('C:/RecoSys/Data/movie.csv', encoding='utf-8')




recommender(5, 10)












