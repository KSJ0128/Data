#0. 패키지 import
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

#1. Data 세팅 및 로드
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

#2. 모델 로드(데이터 마이닝 기법)
clf = QuadraticDiscriminantAnalysis()

#3. 모델 훈련
clf.fit(X, y)

#4. 모델 테스트(Predict, score)
print(clf.predict([[-0.8, -1]]))