import requests

# 定义要发送的数据
data = {'input_char': "春", 'input_style': "七言绝句"}

# 发送POST请求
response = requests.post('http://localhost:5000/lstm_predict', json=data)
# 打印预测结果
print('预测结果:', response.json()['output'])

# 发送POST请求
response = requests.post('http://localhost:5000/transformer_predict', json=data)
# 打印预测结果
print('预测结果:', response.json()['output'])