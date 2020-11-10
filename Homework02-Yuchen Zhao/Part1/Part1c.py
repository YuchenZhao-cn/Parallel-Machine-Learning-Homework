import multiprocessing as mp

list_c = [[2, 3, 4, 5], [6, 9, 10, 12], [11, 12, 13, 14], [21, 24, 25, 26]]


def func(list_1):
    minimum = min(list_1)
    maximum = max(list_1)
    return [(i - minimum)/(maximum-minimum) for i in list_1]


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(func, args=(l1,)) for l1 in list_c]
    pool.close()
    print(results[:10])
