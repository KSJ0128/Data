import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#1. Data 세팅 및 로드
# 학년 11, 키 15, 몸무게 16, 수축 23, 이완 24
x = open("student_health_3.csv","r", encoding="ISO-8859-1");
x1 = csv.reader(x)

row1 = []
row2 = []
row3 = []
row4 = []
row5 = []

for row in x1:
    row1.append(row[11]) # 학년
    row2.append(row[15]) # 키
    row3.append(row[16]) # 몸무게
    row4.append(row[23]) # 수축기
    row5.append(row[24]) # 이완기

x.close()

row_s = []
row_n = []
for r in range(0,len(row1)):
        row_n.append(float(row1[r]))
        row_s.append([float(row2[r]), float(row3[r]),float(row4[r]), float(row5[r])])


#2. 모델 로드(데이터 마이닝 기법), 3. 모델 훈련
clf = KMeans(n_clusters = 2).fit(row_s,row_n)


#4. 모델 테스트(Predict, score)
print(clf.predict([[115, 22, 80, 60]]))
print(clf.predict([[140, 35, 100, 70]]))

