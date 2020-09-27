# -*- coding: utf-8 -*- 

# test and vertify file

# from ekmapTK import line_count as lc
# from ekmapTK import filter
from ekmapTK import read_REFIT
from ekmapTK import read_EKfile
from ekmapTK import do_plot

# from numpy import power
# from numpy import pi
# from numpy import int8
# from numpy import linspace


# from time import time
# from multiprocessing import Pool, Process, Manager
# from multiprocessing import Queue
# from tqdm import tqdm



if __name__ == "__main__":

    house_number = 7
    slice = 6
    file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'
    data2, appQ = read_REFIT(file_path, slice=slice)
    # file_path = 'REFIT/EKMap_House' +  str(house_number) + '.csv'
    # data2, appQ = read_EKfile(file_path)

    # do_plot(data2, appQ, title = 'dfsfs')

    k = 0
    for data3 in data2:
        k += 1
        titl = str(k) + r'in' + str(slice)
        do_plot(data3, appQ, title = titl, do_show=False, 
                fig_type=(r'_' + titl+'.png', r'_' + titl + '.eps', ), )


