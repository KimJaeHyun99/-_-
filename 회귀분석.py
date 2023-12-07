# 라이브러리 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib 글씨체 변경
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 데이터 불러오기
data = pd.read_csv('../data/boston_house/house_price.csv')

# 데이터 쉐입
data.shape

# 데이터 컬럼 확인
data.columns

# 데이터 상위 항목 확인
data.head(10)

data.describe()

data.info()

# 학습 데이터 분리
x = data.drop(['MEDV'], axis = 1)
y = data['MEDV']

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)
print('xtrain 쉐입: ', xtrain.shape)
print('xtest 쉐입: ', xtest.shape)
print('ytrain 쉐입: ', ytrain.shape)
print('ytest 쉐입: ', ytest.shape)

# 훈련 모델에 다중 선형 회귀 모델을 적용시키기
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# 테스트 결과 예측
y_pred = regressor.predict(xtest)

plt.scatter(ytest, y_pred, c='green')
plt.xlabel("가격: 천 달러 단위($1000)")
plt.ylabel("예측된 값")
plt.title("실제 값 vs 예측 값 : 선형 회귀")
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

MSE = mean_squared_error(ytest, y_pred)
MAE = mean_absolute_error(ytest, y_pred)
print("평균 제곱 오차: ", MSE)
print("평균 절대 오차: ", MAE)
print("정확도: ", 100-MSE)
