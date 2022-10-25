# Created or modifed on Oct 2022
# Author: 임일
# Factorizagion Machines(FM) 연습문제 7-1-1 (EachMovie)

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')




K = 250
fm1 = FM(num_x, K, data, y, alpha=0.0007, beta=0.003, train_ratio=0.75, iterations=900, tolerance=0.0005, l2_reg=True, verbose=True)
result = fm1.test()



