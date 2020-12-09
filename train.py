import argparse
import json
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

trainStr = []
trainValue = []

def readData(file):
    f = open(file,'r',encoding='utf-8')
    data = json.load(f)
    for i in data:
        trainStr.append(i['data'])
        trainValue.append(i['label'])
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training_data', type=str, default='train.json')
    args = parser.parse_args()
    trainingData = args.training_data

    readData(trainingData)
    vectorizer = CountVectorizer(max_features=20)
    print(vectorizer)

    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(trainStr))
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    x_train_weight = tf_idf.toarray()

    # 对测试集进行tf-idf权重计算


    print('输出x_train文本向量：')
    print(x_train_weight)
