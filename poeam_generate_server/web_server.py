from poem_lstm.poem_LSTM import *
from poem_transformer.poem_Transformer import *
from flask import Flask, request, jsonify

app = Flask(__name__)

# 定义API端点
@app.route('/lstm_predict', methods=['POST'])
def lstm_predict():
    data = request.json  # 获取POST请求中的JSON数据
    input_char = data['input_char']     # 提取输入字符值
    input_style = data['input_style']   # 提取输入类型值

    while True:
        # 使用模型进行预测
        poem = gen_lstm_poem(input_char)
        length = len(poem.split('，')[0])
        if input_style == "五言绝句" and length == 5:
            output = poem
            break
        if input_style == "七言绝句" and length == 7:
            output = poem
            break
    # 返回预测结果
    return jsonify({'output': output})

# 定义API端点
@app.route('/transformer_predict', methods=['POST'])
def transformer_predict():
    data = request.json  # 获取POST请求中的JSON数据
    input_char = data['input_char']     # 提取输入字符值
    input_style = data['input_style']   # 提取输入类型值

    while True:
        # 使用模型进行预测
        poem = gen_transformer_poem(input_char, input_style)
        length = len(poem)
        print(poem)
        if input_style == "五言绝句" and length == 5:
            output = poem
            break
        if input_style == "七言绝句" and length == 7:
            output = poem
            break

    # 返回预测结果
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)