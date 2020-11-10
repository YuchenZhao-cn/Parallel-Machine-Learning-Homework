import numpy as np
import pandas as pd
import os 
import time
from sklearn import svm
import multiprocessing as mp
from multiprocessing import Pool

def cvkfold(X, y, tuning_params, partitions, k):
    n_tuning_params = tuning_params.shape[0]

    partition = partitions[k]
    Train = np.delete(np.arange(0, X.shape[0]), partition)
    Test = partition
    X_train = X.iloc[Train, :]
    y_train = y.iloc[Train]

    X_test = X.iloc[Test, :]
    y_test = y.iloc[Test]

    accuracies = np.zeros(n_tuning_params)
    for i in range(0, n_tuning_params):
        svc = svm.SVC(C = tuning_params[i], kernel = "linear")
        accuracies[i] = svc.fit(X_train, y_train).score(X_test, y_test)
    return accuracies

def cvkfold1(kk):
    return cvkfold(X, y, tuning_params, partitions, kk)

train = pd.read_csv("1.csv")
# train = train.loc[0: 5000]
fe = ["help", "need", "make", "vote", "donat", "support", "get", "peopl", "trump", "right", "fight", "take", "time",
          "like"
        , "take", "campaign", "elect", "work", "one", "senat", "today", "class5afxspan", "class58cl", "5afzspanspan",
          "new"
        , "span", "state", "day", "back", "protect", "republican", "go", "year", "know", "plea", "care", "join", "everi"
        , "class58cnspan", "democrat", "presid", "voter", "classimg", "way", "court", "want", "sign", "famili"
        , "polit", "run", "stop", "win", "stand", "class5mfr", "47e3img", "free", "would", "give", "american", "put",
          "kavanaugh"
        , "congress"]

X = train[fe]
y = train["political_probability"]

K = 5
tuning_params = np.logspace(-6, -1, 10)
partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), K)
data_list = (0, 1, 2, 3, 4,)

if __name__ == "__main__":

    # Serial
    t0 = time.time()
    for k in range(0, K):
        Accuracies = cvkfold(X, y, tuning_params, partitions, k)

    print('Serial runs %0.3f seconds.' % (time.time() - t0))
    print(Accuracies)

    # 2 CPUs
    t1 = time.time()

    pool1 = mp.Pool(processes=2)
    Accuracies = [pool1.map(cvkfold1, data_list)]
    pool1.close()
    pool1.join()

    print('Pool(2) runs %0.3f seconds.' % (time.time() - t1))
    print(Accuracies)

    # 4 CPUs
    t2 = time.time()

    pool1 = mp.Pool(processes=4)
    Accuracies = [pool1.map(cvkfold1, data_list)]
    pool1.close()
    pool1.join()

    print('Pool(4) runs %0.3f seconds.' % (time.time() - t2))
    print(Accuracies)

    # 8 CPUs
    t3 = time.time()

    pool1 = mp.Pool(processes=8)
    Accuracies = [pool1.map(cvkfold1, data_list)]
    pool1.close()
    pool1.join()

    print('Pool(8) runs %0.3f seconds.' % (time.time() - t3))
    print(Accuracies)

