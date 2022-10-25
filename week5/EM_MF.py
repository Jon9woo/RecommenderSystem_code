# Created or modified on Sep 2022
# author: 임일
# Matrix factorization 2 - Train/Test 분리해서 정확도 계산 (EachMovie)

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')

# New MF class for training & testing
class NEW_MF():







