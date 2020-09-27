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

# import matplotlib.pyplot as plt
import matplotlib.patches as pat
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
    
    highL = []  # rectangle to high light
    for wid, x in zip((2, 4, 4, 4, 2), (0, 6, 14, 22, 30)):
        highL.append(pat.Rectangle((x, 2), wid, 4, 
                                fill=False, edgecolor='c', lw=6))
        highL.append(pat.Rectangle((x, 10), wid, 4, 
                                fill=False, edgecolor='c', lw=6))
    # do_plot(data2, appQ, do_show=True, pats=highL, 
    #         title='1in1_app8 is on')


    k = 0
    for data3 in data2:
        k += 1
        titl = str(k) + r'in' + str(slice) + '_app3 is on and app8 is off'
        do_plot(data3, appQ, title = titl, do_show=False, pats=highL.copy(), 
                fig_type=(r'_' + titl+'.png', r'_' + titl + '.eps', ), )

