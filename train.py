import argparse
import json
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

trainStr = []
trainValue = []

def readData(file):
    f = open(file,'r',encoding='utf-8')
    data = json.load(f)
    for i in data:
        trainStr.append(i['data'])
        trainValue.append(i['label'])
    f.close()


def findParams(text_clf):
    param_total = {
        'vect__ngram_range': [(1, 2)],
        'clf-svm__kernel': ['sigmoid'],
        'clf-svm__nu': [0.42,0.41,0.40,0.39,0.38,0.37],
    }
    with open("D://training.txt","a+",encoding='utf-8')as f:
        print("===============Test NU SIG==================")
        f.write("===============Test NU==================\n")
        gs_clf_svm = GridSearchCV(text_clf, param_total, n_jobs=-1)
        gs_clf_svm = gs_clf_svm.fit(trainStr, trainValue)
        print("results:")
        print(gs_clf_svm.cv_results_)
        print("best scores:")
        print(gs_clf_svm.best_score_)
        print("best params:")
        print(gs_clf_svm.best_params_)
        try:
            f.write("results:\n")
            f.write(str(gs_clf_svm.cv_results_)+"\n")
            f.write("best score:\n")
            f.write(str(gs_clf_svm.best_score_)+"\n")
            f.write("best params:\n")
            f.write(str(gs_clf_svm.best_params_)+"\n")
        except TypeError:
            pass


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

    # text_clf= Pipeline([('vect', CountVectorizer()),
    #                     ('tfidf', TfidfTransformer()),
    #                     ('clf-svm', NuSVC()),
    # ])
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf-svm', NuSVC(kernel='sigmoid',nu=0.35)),
                         ])

    # findParams(text_clf)
    #recent = {}
    # text_clf.fit(trainStr,trainValue)
    # #print(text_clf)
    # joblib.dump(text_clf,'model')
    # param1 = {
    #     'vect__ngram_range':[(1,1),(1,2)],
    #     'tfidf__use_idf': (True, False),
    #     'clf-svm__nu':[0.5,0.6,0.4],
    #     'clf-svm__kernel':['poly'],
    #     'clf-svm__degree':[3,3.5,2.5],
    #     'clf-svm__coef0':[0,0.2,-0.2],
    #     #'clf-svm__class_weight':[None,'balanced'],
    #     'clf-svm__decision_function_shape':['ovo','ovr'],
    # }

    param_total2 = {
        'clf-svm__kernel': ['rbf'],
        'clf-svm__nu':[0.5,0.55,0.45,0.51,0.49],
    }
    param1 = [
        {
        #'vect__ngram_range':[(1,1),(1,2)],
        #'tfidf__use_idf': (True, False),
        'clf-svm__nu':[0.5,0.55,0.45],
        'clf-svm__kernel':['poly'],
        'clf-svm__degree':[3,3.5,2.5],
        #'clf-svm__coef0':[0,0.2,-0.2],
        #'clf-svm__class_weight':[None,'balanced'],
        #'clf-svm__decision_function_shape':['ovo','ovr'],
        },{
            #'vect__ngram_range': [(1, 1), (1, 2)],
            # 'tfidf__use_idf': (True, False),
            #'clf-svm__nu': [0.5, 0.6, 0.4],
            'clf-svm__kernel': ['poly'],
            #'clf-svm__degree': [3, 3.5, 2.5],
             'clf-svm__coef0':[0,0.2,-0.2],
             'clf-svm__class_weight':[None,'balanced'],
            #'clf-svm__decision_function_shape': ['ovo', 'ovr'],
        },{
            'vect__ngram_range': [(1, 1), (1, 2)],
            # 'tfidf__use_idf': (True, False),
            #'clf-svm__nu': [0.5, 0.6, 0.4],
            'clf-svm__kernel': ['poly'],
            #'clf-svm__degree': [3, 3.5, 2.5],
            # 'clf-svm__coef0':[0,0.2,-0.2],
            # 'clf-svm__class_weight':[None,'balanced'],
            'clf-svm__decision_function_shape': ['ovo', 'ovr'],
        },



    ]
    param2 = {
        #'vect__ngram_range':[(1,1),(1,2)],
        #'tfidf__use_idf': (True, False),
        'clf-svm__nu':[0.5,0.55,0.45],
        'clf-svm__kernel':['rbf'],
        #'clf-svm__class_weight':[None,'balanced'],
        #'clf-svm__decision_function_shape':['ovo','ovr'],
    }
    param3 = {
        #'vect__ngram_range':[(1,1),(1,2)],
        #'tfidf__use_idf': (True, False),
        'clf-svm__nu':[0.5,0.55,0.45],
        'clf-svm__kernel':['sigmoid'],
        'clf-svm__coef0':[0,0.05,-0.05],
        #'clf-svm__class_weight':[None,'balanced'],
        #'clf-svm__decision_function_shape':['ovo','ovr'],
    }


    #text_clf.fit(training,training_val)




