# 입력 데이터와 다항식을 이진으로 표현
multi = '100101'
data = '101101001'

# 데이터와 다항식을 이진 문자열로 변경
multi_binary = int(multi, 2)
data_binary = int(data, 2)

# 체크섬 계산
checksum = data_binary % multi_binary

# 결과 출력
print(f"Checksum: {bin(checksum)[2:]}")  # 결과를 이진으로 출력
