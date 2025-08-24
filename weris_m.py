#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json # json 라이브러리 추가

# 데이터 전처리: 결측치 제거
# 이 부분은 현재 코드에서 사용되지 않으므로, 원래 코드의 맥락을 유지했습니다.
# df_cleaned = df.dropna(axis=1)

# 증상 데이터 파일을 읽어옵니다.
try:
    data1 = pd.read_csv('data1.csv')
    data2 = pd.read_csv('data2.csv')
    data3 = pd.read_csv('data3.csv')
except FileNotFoundError:
    print("오류: data1.csv, data2.csv, data3.csv 중 하나 이상을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

# =========================================================================
# 💡 사용자 입력 부분 💡
# =========================================================================

# JSON 파일에서 증상 데이터를 읽어옵니다.
try:
    with open('symptom_data.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 'week' 키를 제외하고 증상 데이터를 딕셔너리로 변환
    symptom_data = {
        key: 't' if value == 1 else 'f'
        for key, value in json_data.items()
        if key != 'week'
    }

    # 주차 정보를 JSON에서 읽어옵니다.
    week = json_data.get('week')
    if week is None or not isinstance(week, int) or week < 1:
        print("오류: JSON 파일에 유효한 'week' 정보가 없습니다. 1보다 큰 정수를 사용해주세요.")
        exit()

except FileNotFoundError:
    print("오류: symptom_data.json 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()
except json.JSONDecodeError:
    print("오류: symptom_data.json 파일이 유효한 JSON 형식이 아닙니다.")
    exit()

print(f"\n입력받은 주차: {week}")
print(f"입력받은 증상 데이터: {symptom_data}")

# =========================================================================
# 스트레스 지수 계산 및 JSON 저장
# =========================================================================

scores = []
for df_stress in [data1, data2, data3]:
    score = 0
    for col, user_val in symptom_data.items():
        try:
            csv_val = df_stress[col].iloc[0]
            if user_val == 't':
                score += 1.3 if csv_val == 1 else 0.8
            elif user_val == 'f':
                score += 0.3 if csv_val == 1 else 0.0
        except (KeyError, IndexError):
            continue
    scores.append(score)

stress_level = min(np.mean(scores), 10)
print(f"\n계산된 스트레스 지수: {stress_level:.2f}")

# 계산된 스트레스 지수를 JSON 파일로 저장합니다.
result = {
    "week": week,
    "stress_level": round(stress_level, 2)
}

try:
    with open('stress_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print("\n✅ 스트레스 지수가 'stress_result.json' 파일에 성공적으로 저장되었습니다.")
except IOError:
    print("\n❌ 오류: JSON 파일을 저장하는 데 실패했습니다. 쓰기 권한을 확인해주세요.")


# In[ ]:




