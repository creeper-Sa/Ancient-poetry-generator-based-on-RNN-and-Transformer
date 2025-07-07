# 🌸 跨越千年，赋诗成文：用 LSTM 和 Transformer 重现古诗的韵律与情感

### ✨ 前言

在深度学习大量古诗词之后，AI 赋诗如下：  

> <img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/03e81a45-df21-4b88-97f5-0de04b2757f6" />

> 参考数据集来源：  
> 🔗 [Chinese-Poetry 数据集](https://github.com/chinese-poetry/chinese-poetry)

---

### 🖥️ 设备信息

- **设备型号**：Lenovo Legion R7000P ADR10

---

### 🧰 环境配置

在开始之前，请确保你已安装以下 Python 依赖：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

### 📚 模型训练与体验

#### ✅ 训练模型 - RNN和Transformer

<u>推荐使用pycharm直接启动后端项目</u>

运行以下命令以启动模型训练过程：

```bash
python train.py
```

模型将基于古诗数据集进行训练，模拟传统诗词的语言风格与格律。

#### 🚀 快速体验

若你想快速体验 AI 作诗功能：

```bash
python test.py
```

无需从零开始训练，我们已提供预训练好的模型参数。只需克隆整个项目，即可直接使用。



我们也对比了⽂本⽣成 PPL（困惑度）低于 100（RNN 架构）；Transformer 架构 PPL 对⽐分析。

RNN架构:

<img width="500" height="200" alt="Image" src="https://github.com/user-attachments/assets/03d9e6c8-17c3-430c-821f-57f732b9aa35" />

Transformer架构:

<img width="500" height="200" alt="Image" src="https://github.com/user-attachments/assets/c0de3e09-d357-4848-8546-28450bb74ee3" />

---

### 🌐 可视化交互界面

控制台太枯燥？没关系，我们为你准备了 **Vue3网页版界面** 采取前后端分离的方式：

```bash
npm i       //安装相关依赖
npm run dev //启动服务
```

运行后，控制台会输出一个本地访问地址。点击该链接，即可体验交互式的 AI 诗歌生成器。

你还可以将项目部署至服务器，与他人一同领略智能赋诗的魅力。



---

### 📌 项目亮点总结

- 基于 **LSTM和Transformer 循环神经网络** 实现古诗生成
- 使用 **Vue** 打造可交互的 Web 界面
- 内置预训练模型，快速上手
- 支持继续训练与自定义数据集拓展

---
