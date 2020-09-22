# -*- coding: utf-8 -*- 

# test and vertify file

# from ekmapTK import line_count as lc
# from ekmapTK import filter
from ekmapTK import TOTAL_LINE, data_read as dr
from ekmapTK import beauty_time as bt

from numpy import sqrt
from numpy import pi

from time import time
from multiprocessing import Pool
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
def do1(x):
    x = x+515000000
    y = x**pi
    return int(''.join(list(str(x))))


if __name__ == "__main__":
    tic = time()
    TOTAL_LINE = 5000000
    with tqdm(total = TOTAL_LINE, leave = False, ascii = True, 
            bar_format = "Loading House1...{l_bar}{bar}|{n_fmt}/{total_fmt}") as pybar:

        for x in range(TOTAL_LINE):
            x = x+515000000
            y = x**pi
            z = y**pi
            z = int(''.join(list(str(int(z)))))
            pybar.update(1)

    toc = time()
    print(bt(toc-tic))

