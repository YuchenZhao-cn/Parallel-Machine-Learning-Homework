import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from multiprocessing import Pool
from sklearn import svm

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


def cvkfold(X, y, tuning_params, partitions, k):
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
    return accuracies

K = 5
tuning_params = np.logspace(-6, -1, 10)
partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), K)
data_list = (0, 1, 2, 3, 4,)


def cvkfold1(kk):
    return cvkfold(X, y, tuning_params, partitions, kk)


if __name__ == "__main__":
    # serial
    t0 = time.time()

    for k in range(0, K):
        Accuracies = cvkfold(X, y, tuning_params, partitions, k)

    print('Serial runs %0.3f seconds.' % (time.time() - t0))

    # CV_accuracy = np.mean(Accuracies, axis=0)
    # best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    # print('Best tuning param %0.6f.' % best_tuning_param)

    # 2 processes
    t1 = time.time()

    pool1 = mp.Pool(processes=2)
    Accuracies = [pool1.map(cvkfold1, data_list)]
    pool1.close()
    pool1.join()

    print('Pool(2) runs %0.3f seconds.' % (time.time() - t1))


    # CV_accuracy = np.mean(Accuracies, axis=0)
    # best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    # print('Best tuning param %0.6f.' % best_tuning_param)

    # 4 processes
    t2 = time.time()

    pool2 = mp.Pool(processes=4)
    Accuracies = [pool2.map(cvkfold1, data_list)]
    pool2.close()
    pool2.join()

    print('Pool(4) runs %0.3f seconds.' % (time.time() - t2))

    # CV_accuracy = np.mean(Accuracies, axis=0)
    # best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    # print('Best tuning param %0.6f.' % best_tuning_param)

    # 8 processes
    t3 = time.time()

    pool3 = mp.Pool(processes=8)
    Accuracies = [pool3.map(cvkfold1, data_list)]
    pool3.close()
    pool3.join()

    print('Pool(8) runs %0.3f seconds.' % (time.time() - t3))

    # CV_accuracy = np.mean(Accuracies, axis=0)
    # best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
    # print('Best tuning param %0.6f.' % best_tuning_param)
