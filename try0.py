# -*- coding: utf-8 -*- 

# test and vertify file

# from ekmapTK import line_count as lc
# from ekmapTK import filter
from ekmapTK import TOTAL_LINE, data_read as dr
from ekmapTK import beauty_time as bt

from numpy import sqrt
from numpy import pi
from numpy import int8
from numpy import linspace

from time import time
from multiprocessing import Pool
from multiprocessing import Queue
from tqdm import tqdm

wt = 50      # width per part taken
itl = 1/wt   # interval  is 1/50


# import dataset, REFIT in this case
# and reformat
# f = 'REFIT/CLEAN_House1.csv'
# data2 = dr(f)
# # total_line = data2[-1]
# # print(total_line)

# t = '='*6
# print(t + 'done' + t)


# try multiprocessing

val = {}

def do1(x, val):

    x = x+515000000
    z = x**pi
    y = int8(x)
    if y in val.keys():
        val[y] += 1
    else:
        val[y] = 1
    int(''.join(list(str(int(z)))))

    return val

def do2(x):
    val = {}
    for k in range(x[0], x[1]):
        val = do1(k, val)
    
    return val
    

if __name__ == "__main__":
    qu = Queue()

    tic = time()
    TOTAL_LINE = 5000000
    x1 = linspace(0, TOTAL_LINE, num = 20, dtype = 'int')
    x2 = ((x1[k], x1[k+1]) for k in range(len(x1)-1))
    # x2 is a generator

    pool = Pool()
    
    result = pool.map(do1, x2)
    pool.close()
    pool.join()

    linspace()
    toc = time()
    print(bt(toc-tic))
    print(val.items())
    print(f'{result[0]=}')

