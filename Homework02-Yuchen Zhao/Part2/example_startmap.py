# Solution Without Paralleization
import numpy as np
from time import time
import multiprocessing as mp

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[100000, 1000])
data = arr.tolist()
#data[:5]


def examp01(row, minimum, maximum):
    """Returns how many numbers between `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(examp01, [(row, 4, 8) for row in data])
    pool.close()
    print('Serial execution:', results[:10])

# results = []
# for row in data:
#     results.append(examp01(row, minimum=4, maximum=8))
#
# print('Serial execution:', results[:10])

