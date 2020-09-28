# -*- coding: utf-8 -*- 

# test and vertify file

# from ekmapTK import line_count as lc
# from ekmapTK import filter
from numpy.lib.nanfunctions import nanmean
from ekmapTK import read_REFIT, slice_REFIT
from ekmapTK import read_EKfile
from ekmapTK import do_plot

# from numpy import power
# from numpy import pi
# from numpy import int8
# from numpy import linspace

# import matplotlib.pyplot as plt
import matplotlib.patches as pat
# from time import time
# from multiprocessing import Pool, Process, Manager
# from multiprocessing import Queue
# from tqdm import tqdm

def do1(house_number, slice):
    """
    generate EKMap slices with rectangle highlight
    """

    file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'
    data2, appQ = read_REFIT(file_path, slice=slice)
    # file_path = 'REFIT/EKMap_House' +  str(house_number) + '.csv'
    # data2, appQ = read_EKfile(file_path)

    highL = []  # rectangle to high light
    for wid, x in zip((2, 4, 4, 4, 2), (0, 6, 14, 22, 30)):
        # highL.append(pat.Rectangle((x, 2), wid, 4, 
        #                         fill=False, edgecolor='c', lw=6))
        # highL.append(pat.Rectangle((x, 10), wid, 4, 
        #                         fill=False, edgecolor='c', lw=6))
        highL.append(pat.Rectangle((x, 0), wid, 16, 
                                fill=False, edgecolor='c', lw=6))
        pass
    do_plot(data2, appQ, do_show=True, pats=highL, 
            title='1in1_app8 is on')


    k = 0
    for data3 in data2:
        k += 1
        titl = str(k) + r'in' + str(slice) + '_app3 is on and app8 is off'
        do_plot(data3, appQ, title = titl, do_show=True, pats=highL.copy(), 
                fig_type=(r'_' + titl+'.png', r'_' + titl + '.eps', ), )

    return None


def do2(house_number, slice):
    """
    generate data fits Mingjun Zhong's code
    """

    file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'

    for ind in range(int(slice/2)):
    # for ind in (0, ):
        n_t = ind + 1
        n_v = ind + 2
        slice_REFIT(file_path=file_path, n_slice=slice, 
                    n_valid=n_v, n_test=n_t, n_app=3, 
                    save_dir = './ex' + str(ind) + '/freezer/', )
        print('-'*20)
        pass

    return None


if __name__ == "__main__":

    house_number = 7
    slice = 10

    do2(house_number, slice)

