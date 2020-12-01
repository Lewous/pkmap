# -*- coding: utf-8 -*-

# test and vertify file

# from ekmapTK import line_count as lc
# from ekmapTK import filter
from ekmapTK import read_REFIT, slice_REFIT
from ekmapTK import read_EKfile, GC
from ekmapTK import do_plot, do_plot2

# from numpy import power
# from numpy import pi
# from numpy import int8
# from numpy import linspace
from numpy import load
from numpy import arange, log
from numpy import random as rd

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

# pre-set pats data: x_data, y_data, width, hight
pat_data = {
    '378': ((0.5, 16.5), (1.5, 9.5), 14, 4),
    '-3-7-8': ((-2.5, 13.5, 29.5), (-2.5, 5.5, 13.5), 4, 4),
    '-3-4-7-8': ((-1.5, 14.5, 30.5), (-1.5, 6.5, 14.5), 2, 2),
    '489': ((0.5,8.5,16.5,24.5), (0.5,4.5,8.5,12.5),6,2)
}

def GAD(n):
    """
    create AD with n variables
    """
    AD = {}
    for ky in GC(n):
        val = int(rd.random()*10**rd.randint(2,7))
        if (-1)**rd.randint(0,3) > 0:
            AD[ky] = val
        else:
            AD[ky] = 0
    return AD


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

    for x, y in zip((0.5, 0.5, 16.5, 16.5), (1.5, 9.5, 1.5, 9.5)):
        # highL.append(pat.Rectangle((x, y), 6, 4,
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

    return do_plot(data2, (0,), titles=tuple(str(k+1) + r'in' + str(slice) for k in range(slice)),
            do_show=True, pats=highL, fig_types=('in' + str(slice) + '.eps', ),
            )


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

    return print('-'*20)


def do3(house_number, titles='', slice = '', **kwargs):
    """
    do basic single plot
    """
    file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'
    data2 = read_REFIT(file_path, save_file=True, slice=slice)

    # do12(data2, '-3-7-8', fig_types=('_1.eps','_1.png',),
    #     do_show=False, 
    #     # titles='House'+str(house_number), 
    #     **kwargs)
    # do12(data2, '-3-4-7-8', fig_types=('_3.eps','_3.png',),
    #     do_show=False, 
    #     # titles='House'+str(house_number), 
    #     **kwargs)
    # do12(data2, 378, fig_types=('_2.eps','_2.png',),
    #             do_show=False, 
    #             # titles='House'+str(house_number), 
    #             **kwargs)

    return 0
    

def do12(data, p2hs, cols = ('c', 'b', 'k'), **kwargs):
    '''
    plot 9 varibales' WKMap with pats
    data: data to plot
    p2hs: int, str, iterable (3 max) are accepted

    return:
    '''

    if isinstance(p2hs, (str, int)):
        p2hs = (str(p2hs),)
    elif isinstance(p2hs, (list, tuple)):
        p2hs = tuple([str(x) for x in p2hs])
    else:
        p2hs = ()       # empty tuple

    highL=[]
    ofst = 0.6      # offset
    for p2h, col, ofst in zip(p2hs, 
                             cols, 
                             (0.2, 0.4, 0.6)):
        if not p2h:
            # is empty or None or ...
            continue

        if p2h in pat_data.keys():
            xd, yd, wid, hit = pat_data[p2h]
        else:
            xd, yd = (0,), (0,)
            wid, hit = 4, 4
        for x, y in ((u,v) for u in xd for v in yd):
            highL.append(pat.FancyBboxPatch(
                        (x+ofst, y+ofst), wid-ofst*2, hit-ofst*2,
                        fill=False, color=col, lw=6))

    return do_plot2(data, pats=highL, **kwargs)


if __name__ == "__main__":

    house_number = 17
    slice = 9
    n_app = 3
    # house_number = (1, 2, 3, 4, 5, 
    #                 6, 7, 8, 9, 10,
    #                 11, 12, 13, 15, 16, 
    #                 17, 18, 19, 20, 21, )
    do3(house_number, slice = slice)
    # do2(house_number, slice, n_app)

    # for k in (1, 2, 3, 4, 5, 
    #           6, 7, 8, 9, 10,
    #           11, 12, 13, 15, 16, 
    #           17, 18, 19, 20, 21, ):
    #     do3(k)

    # do3(house_number)
    # AD = GAD(9)
    # print(AD)
    p2h =''    # pats to hightlight
    # do12(AD, p2h, )
    # do_plot2(AD, fig_types=('.png',), pats=highL, do_show=True)
    # # print(f'{sum(AD.values())}')