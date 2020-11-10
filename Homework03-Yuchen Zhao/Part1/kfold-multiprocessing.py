import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Process
from sklearn import svm
import time

test = np.loadtxt("optdigits.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]

# Plot some of the digits
# fig = plt.figure(figsize=(8, 6))
# fig.tight_layout()
# for i in range(0, 20):
#     ax = fig.add_subplot(5, 5, i + 1)
#     ax.imshow(X[i].reshape((8,8)), cmap = "Greys", vmin = 0, vmax = 16)
# plt.show()


def cvkfold(X, y, tuning_params, partitions, k, q):
    n_tuning_params = tuning_params.shape[0]

    partition = partitions[k]
    Train = np.delete(np.arange(0, X.shape[0]), partition)
    Test = partition
    X_train = X[Train, :]
    y_train = y[Train]

    X_test = X[Test, :]
    y_test = y[Test]

    accuracies = np.zeros(n_tuning_params)
    for i in range(0, n_tuning_params):
        svc = svm.SVC(C = tuning_params[i], kernel = "linear")
        accuracies[i] = svc.fit(X_train, y_train).score(X_test, y_test)
    q.put(accuracies)


def cvkfold1(X, y, tuning_params, partitions, k, q, count):
    for kk in range(k, k + count):
        cvkfold(X, y, tuning_params, partitions, kk, q)

K = 5
tuning_params = np.logspace(-6, -1, 10)
partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), K)

if __name__ == "__main__":
    # serial
    t1 = time.time()

    q = mp.Queue()
    p_list = []

    p = Process(target=cvkfold1, args=(X, y, tuning_params, partitions, 0, q, 5,))
    p.start()
    p_list.append(p)

    for p in p_list:
        p.join()

    Accuracies = [q.get() for i in p_list]

    print('Serial runs %0.3f seconds.' % (time.time() - t1))

    CV_accuracy = np.mean(Accuracies, axis=0)
    best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    print('Best tuning param %0.6f.' % best_tuning_param)

    # 2 processes
    t1 = time.time()

    q = mp.Queue()
    p_list = []

    p1 = Process(target=cvkfold1, args=(X, y, tuning_params, partitions, 0, q, 2,))
    p1.start()
    p_list.append(p1)

    p2 = Process(target=cvkfold1, args=(X, y, tuning_params, partitions, 2, q, 3,))
    p2.start()
    p_list.append(p2)

    for p in p_list:
        p.join()

    Accuracies = [q.get() for i in p_list]

    print('Process(2) runs %0.3f seconds.' % (time.time() - t1))

    CV_accuracy = np.mean(Accuracies, axis = 0)
    best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    print('Best tuning param %0.6f.'% best_tuning_param)

    # 4 processes
    t2 = time.time()

    q = mp.Queue()
    p_list = []

    p1 = Process(target=cvkfold1, args=(X, y, tuning_params, partitions, 0, q, 2,))
    p1.start()
    p_list.append(p1)

    for k in range(2, K):
        p2 = Process(target=cvkfold, args=(X, y, tuning_params, partitions, k, q,))
        p2.start()
        p_list.append(p2)

    for p in p_list:
        p.join()

    Accuracies = [q.get() for i in p_list]

    print('Process(4) runs %0.3f seconds.' % (time.time() - t2))

    CV_accuracy = np.mean(Accuracies, axis=0)
    best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    print('Best tuning param %0.6f.' % best_tuning_param)

    # 8 processes
    t3 = time.time()

    q = mp.Queue()
    p_list = []

    for k in range(0, K):
        p = Process(target=cvkfold, args=(X, y, tuning_params, partitions, k, q,))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    Accuracies = [q.get() for i in p_list]

    print('Process(8) runs %0.3f seconds.' % (time.time() - t3))

    CV_accuracy = np.mean(Accuracies, axis=0)
    best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    print('Best tuning param %0.6f.' % best_tuning_param)
