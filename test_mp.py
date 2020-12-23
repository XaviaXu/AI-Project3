import argparse
import joblib
import json
import multiprocessing as mp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVC

core = 8

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_data', type=str, default='testdataexample')
    parser.add_argument('-m', '--model', type=str, default='model')

    args = parser.parse_args()
    testData = args.test_data
    model = args.model


    f = open(testData,'r',encoding='utf-8')
    test = np.array(json.load(f))
    testList = np.array_split(test,core)
    clf = joblib.load(model)

    pool = mp.Pool(core)
    result = []
    for i in range(core):
        result.append(pool.apply_async(clf.predict,args=(testList[i]).tolist(),))
    #res = clf.predict(test)

    f.close()
    f = open("output.txt",'a+',encoding='utf-8')
    for i in result:
        f.write(str(i.get))
        f.write("\n")
    f.close()