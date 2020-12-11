import argparse
import json
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.datasets import fetch_20newsgroups

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


    #
    # twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    # twenty_train.target_names  # prints all the categories
    # print("\n".join(twenty_train.data[0].split("\n")[:3]))

    # text_clf = Pipeline([('vect',CountVectorizer),
    #                      ('tfidf',TfidfTransformer()),
    #                      ('clf-svm',SGDClassifier(loss='hinge'))
    #                      ])
    text_clf= Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge',
                                                       penalty='l2',
                                                        alpha = 1e-3
                                )),
    ])
    training = trainStr[:20000]
    training_val = np.array(trainValue[:20000])
    text_clf.fit(training,training_val)
    predicted = text_clf.predict(trainStr[:-5000])
    print(np.mean(predicted==trainValue[:-5000]))

    # vectorizer = CountVectorizer()
    # trainCounts = vectorizer.fit_transform(trainStr)
    # trainCounts.shape
    # print(vectorizer)
    #
    # tf_idf_transformer = TfidfTransformer()
    # # 将文本转为词频矩阵并计算tf-idf
    # train_tfidf = tf_idf_transformer.fit_transform(trainCounts)
    # print(train_tfidf.shape)
    # # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重


    # 对测试集进行tf-idf权重计算


    print('输出x_train文本向量：')

