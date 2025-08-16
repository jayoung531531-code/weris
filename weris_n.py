#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 기존의 pip 설치 부분은 그대로 둡니다.
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install requests')

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import requests
import pickle

# 'transposed_data.csv' 파일을 읽고, 'abc' 열을 인덱스로 설정합니다.
try:
    df = pd.read_csv('transposed_data.csv', index_col='abc')
except FileNotFoundError:
    print("오류: transposed_data.csv 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

# 데이터 전처리: 결측치 제거
df_cleaned = df.dropna(axis=1)

# 💡 증상 데이터 파일을 한 번에 로드하지 않습니다. 💡
# data1, data2, data3 변수 선언 부분을 제거하고, 필요한 시점에 chunk로 읽습니다.

# 💡 API로부터 데이터를 가져오는 함수 (순수 API 연동) 💡
def get_data_from_api():
    """
    API로부터 증상 및 주차 데이터를 받아오는 역할을 합니다.
    """
    API_FETCH_URL = 'http://api.example.com/symptoms_and_week'
    try:
        response = requests.get(API_FETCH_URL)
        response.raise_for_status()
        api_data = response.json()
        symptom_data = api_data.get('symptoms')
        week = api_data.get('week')
        if not symptom_data or not week:
            print("API 응답에 'symptoms' 또는 'week' 데이터가 없습니다.")
            return None, None
        return symptom_data, week
    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return None, None

# =========================================================================
# 💡 API 연동 함수를 호출하여 데이터 가져오기 💡
# =========================================================================

symptom_data, week = get_data_from_api()

if symptom_data is None or week is None:
    print("API로부터 데이터가 들어오지 않았습니다.")
    exit()

print(f"\nAPI로부터 받은 주차: {week}")
print(f"API로부터 받은 증상 데이터: {symptom_data}")

# =========================================================================
# 스트레스 지수 계산 (메모리 최적화)
# =========================================================================
file_paths = ['data1.csv', 'data2.csv', 'data3.csv']
scores = []

for file_path in file_paths:
    score = 0
    # chunksize를 1000으로 설정하여 1000줄씩 읽습니다.
    # 파일 크기에 따라 이 값을 조정할 수 있습니다.
    try:
        chunk_reader = pd.read_csv(file_path, chunksize=1000)
        for chunk in chunk_reader:
            for col, user_val in symptom_data.items():
                if col in chunk.columns:
                    csv_val = chunk[col].iloc[0]
                    if user_val is True:
                        score += 1.3 if csv_val == 1 else 0.8
                    elif user_val is False:
                        score += 0.3 if csv_val == 1 else 0.0
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
        continue
    except KeyError:
        print(f"오류: {file_path} 파일에 필요한 열이 없습니다. 건너뜁니다.")
        continue
    scores.append(score)

# 모든 파일이 존재하지 않는 경우를 대비
if not scores:
    print("계산할 유효한 데이터가 없습니다. 프로그램을 종료합니다.")
    exit()
    
stress_level = min(np.mean(scores), 10)
print(f"\n계산된 스트레스 지수: {stress_level:.2f}")

# =========================================================================
# 🤖 머신러닝 모델: KNN 분류기 🤖
# =========================================================================
model_filename = 'knn_model.pkl'

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    print(f"\n'{model_filename}' 파일에서 모델을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"\n'{model_filename}' 파일을 찾을 수 없습니다. 새로운 모델을 학습하고 저장합니다.")
    X_train = np.array([
        [1, 5.2], [2, 3.8], [3, 7.5], [4, 6.1], [5, 9.0], [6, 4.5]
    ])
    y_train = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"새로운 모델이 '{model_filename}' 파일로 저장되었습니다.")

# 3. 예측할 새로운 데이터 포인트 준비
new_data_point = np.array([[week, stress_level]])

# 4. 모델을 사용하여 예측
prediction = model.predict(new_data_point)

print(f"\n입력하신 데이터는 '{prediction[0]}' 그룹에 가장 가깝습니다.")

# =========================================================================
# 💡 출력 값을 API로 전송하는 부분 💡
# =========================================================================
API_URL = 'http://api.example.com/results'

payload = {
    'stress_level': float(f"{stress_level:.2f}"),
    'prediction': prediction[0],
    'week': week
}

print(f"\nAPI로 전송할 데이터: {payload}")

try:
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        print("\n✅ 데이터 전송 성공!")
        print(f"서버 응답: {response.json()}")
    else:
        print(f"\n❌ 데이터 전송 실패! 상태 코드: {response.status_code}")
        print(f"서버 응답: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"\n⚠️ API 통신 중 오류 발생: {e}")


# In[ ]:




