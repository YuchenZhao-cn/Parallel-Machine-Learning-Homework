import numpy as np
import pandas as pd
import multiprocessing as mp

df = pd.DataFrame(np.random.randint(3, 10, size=[5, 2]))
print(df.head())


def func(row):
    return round(row[1]**2 + row[2]**2, 2)**0.5


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    result = pool.imap(func, df.itertuples(name=False), chunksize=10)
    output = [round(x, 2) for x in result]
    pool.close()
    print(output)

