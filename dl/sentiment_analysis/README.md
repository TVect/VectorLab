# 任务说明

使用 LSTM 做的情感分类的任务, 准确率 0.81+

暂未单独实现预测代码

# 数据来源

个人的百度网盘: [sentiment_ntu 数据](https://pan.baidu.com/s/1wsLfiCNnxJnnKy6l2ucWuQ)

解压文件到 data 文件夹下(去掉外层目录), 即可.

# 使用说明

```
cd model

python WordRepresentation.py    # 使用gensim, 预训练得词向量表示. 完成之后会生成文件 word2vec.stem.wv

python TfSentiAnalyzer.py       # 情感分类模型训练
```

---

`tensorflow 1.4` `python 3.6` `gensim 2.3.0`
