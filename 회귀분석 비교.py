# 라이브러리 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# matplotlib 글씨체 변경
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
data = pd.read_csv('../data/boston_house/house_price.csv')

# 데이터 쉐입
print("데이터 쉐입:", data.shape)

# 데이터 컬럼 확인
print("데이터 컬럼:", data.columns)

# 데이터 상위 항목 확인
print("상위 항목:\n", data.head(10))

# 데이터 기초 통계 정보 확인
print("기초 통계 정보:\n", data.describe())

# 데이터 정보 확인
print("데이터 정보:\n", data.info())

# 학습 데이터 분리
x = data.drop(['MEDV'], axis=1)
y = data['MEDV']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
print('xtrain 쉐입: ', xtrain.shape)
print('xtest 쉐입: ', xtest.shape)
print('ytrain 쉐입: ', ytrain.shape)
print('ytest 쉐입: ', ytest.shape)

# 직접 구현한 선형 회귀 모델
def custom_linear_regression(x, theta):
    return np.dot(x, theta)

def gradient_descent(x, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        error = np.dot(x, theta) - y
        gradient = np.dot(x.T, error) / m
        theta -= learning_rate * gradient
    return theta

# 행렬에 상수 열 추가
xtrain_custom = np.c_[np.ones(xtrain.shape[0]), xtrain.values]
xtest_custom = np.c_[np.ones(xtest.shape[0]), xtest.values]

# 초기 가중치(theta) 설정
theta_initial = np.zeros(xtrain_custom.shape[1])

# 경사 하강법을 사용하여 theta 훈련
theta_trained = gradient_descent(xtrain_custom, ytrain, theta_initial, learning_rate=0.01, iterations=1000)

# 직접 구현한 모델로 예측
y_pred_custom = custom_linear_regression(xtest_custom, theta_trained)

# Scikit-learn의 Linear Regression 모델 훈련
regressor = make_pipeline(StandardScaler(), LinearRegression())
regressor.fit(xtrain, ytrain)

# Scikit-learn Linear Regression으로 예측
y_pred_sklearn = regressor.predict(xtest)

# 결과 비교를 위한 시각화
plt.scatter(ytest, y_pred_custom, c='blue', label='Custom Linear Regression')
plt.scatter(ytest, y_pred_sklearn, c='green', label='Scikit-learn Linear Regression')
plt.xlabel("실제 가격 ($1000)")
plt.ylabel("예측된 가격 ($1000)")
plt.title("직접 구현 vs Scikit-learn 선형 회귀 비교")
plt.legend()
plt.show()

# 성능 비교
MSE_custom = mean_squared_error(ytest, y_pred_custom)
MSE_sklearn = mean_squared_error(ytest, y_pred_sklearn)

MAE_custom = mean_absolute_error(ytest, y_pred_custom)
MAE_sklearn = mean_absolute_error(ytest, y_pred_sklearn)

print("직접 구현한 선형 회귀 - MSE: {:.2f}, MAE: {:.2f}".format(MSE_custom, MAE_custom))
print("Scikit-learn 선형 회귀 - MSE: {:.2f}, MAE: {:.2f}".format(MSE_sklearn, MAE_sklearn))
