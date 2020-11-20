# -*- coding: utf-8 -*-

# test and vertify file

# from ekmapTK import line_count as lc
# from ekmapTK import filter
from ekmapTK import read_REFIT, slice_REFIT
from ekmapTK import read_EKfile
from ekmapTK import do_plot, GC

# from numpy import power
# from numpy import pi
# from numpy import int8
# from numpy import linspace
from numpy import load
from numpy import arange, log

import matplotlib.pyplot as plt
import matplotlib.patches as pat
# from time import time
from multiprocessing import Pool
# from multiprocessing import Queue
# from tqdm import tqdm
import seaborn as sn
import pandas as pd

# artifical Dataset for illustration propuse
AD = {'0000': 0, 
      '0001': 0, 
      '0011': 0, 
      '0010': 0, 
      '0110': 0, 
      '0111': 0, 
      '0101': 0, 
      '0100': 0, 
      '1100': 255, 
      '1101': 0, 
      '1111': 604, 
      '1110': 58852, 
      '1010': 2771, 
      '1011': 38, 
      '1001': 40272, 
      '1000': 1282001, 
      }


def do1(house_number, slice):
    """
    generate EKMap slices with rectangle highlight
    """

    file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'
    data2 = read_REFIT(file_path, slice=slice)
    # file_path = 'REFIT/EKMap_House' +  str(house_number) + '.csv'
    # data2 = read_EKfile(file_path)

    highL = []  # rectangle to high light
    # highL.append(pat.Rectangle((-0.5, 5.5), 17, 1,
    #                            fill=False, edgecolor='c', lw=3))
    # highL.append(pat.Rectangle((15.5, 5.5), 1, 10,
    #                            fill=False, edgecolor='c', lw=3))

    for wid, x in zip((2, 4, 4, 4, 2), (0, 6, 14, 22, 30)):
        # highL.append(pat.Rectangle((x, 2), wid, 4,
        #                         fill=False, edgecolor='c', lw=6))
        # highL.append(pat.Rectangle((x, 10), wid, 4,
        #                         fill=False, edgecolor='c', lw=6))
        # highL.append(pat.Rectangle((x, 0), wid, 16,
        #                            fill=False, edgecolor='c', lw=6))
        pass

    # for k, datax in zip(range(slice), data2):
    #     titl = str(k+1) + r'in' + str(slice)  # + '_app3 is on and app8 is off'
    #     do_plot((datax, ), appQ, titles=(titl, ), do_show=True, pats=highL, )
        # fig_types=(r'_' + titl+'.png', r'_' + titl + '.eps', ), )
        # fig_types=(r'_' + titl+'.png', ), )

    do_plot(data2, (0,), titles=tuple(str(k+1) + r'in' + str(slice) for k in range(slice)),
            do_show=True, pats=highL, fig_types=('in' + str(slice) + '.eps', ),
            )

    return None


def do2(house_number, slice, n_app):
    """
    generate data fits Mingjun Zhong's trainging and testing
    """

    file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'

    with Pool() as pool:
        pool.map(slice_REFIT, ((file_path, slice, ind+2, ind+1,
                                n_app, './exm' + str(ind) + '/freezer/', )
                               for ind in range(slice-4)))
        pool.close()
        pool.join()

    print('-'*20)
    pass

    return None


def do3(house_number):
    """
    do basic single plot
    """
    file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'
    data2 = read_REFIT(file_path)
    
    do_plot(data2, titles='House'+str(house_number), do_show=False, 
            fig_types=('.eps', ))

if __name__ == "__main__":

    house_number = 15
    slice = 9
    n_app = 3
    # house_number = (1, 2, 3, 4, 5, 
    #                 6, 7, 8, 9, 10,
    #                 11, 12, 13, 15, 16, 
    #                 17, 18, 19, 20, 21, )
    # do1(house_number, slice)
    # do2(house_number, slice, n_app)

    for k in (1, 2, 3, 4, 5, 
              6, 7, 8, 9, 10,
              11, 12, 13, 15, 16, 
              17, 18, 19, 20, 21, ):
        do3(k)
    # do3(house_number)
    # do_plot(AD)
    
