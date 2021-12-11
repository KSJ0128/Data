#0. 패키지 import
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. Data 세팅 및 로드
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

#2. 모델 로드(데이터 마이닝 기법)
clf = LinearDiscriminantAnalysis()

#3. 모델 훈련
clf.fit(X, y)

#4. 모델 테스트(Predict, score)
print(clf.predict([[-0.8, -1]]))

# 결과 = 1 -> X의 0-2번 데이터와 y의 0-2번 데이터(0으로 변경하면 결과 0)와 매칭..?