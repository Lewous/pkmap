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
from multiprocessing import Pool
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
        # highL.append(pat.Rectangle((x, 0), wid, 16,
        #                            fill=False, edgecolor='c', lw=6))
        pass

    # for k, datax in zip(range(slice), data2):
    #     titl = str(k+1) + r'in' + str(slice)  # + '_app3 is on and app8 is off'
    #     do_plot(datax, appQ, title=titl, do_show=False, pats=highL,
    #             # fig_types=(r'_' + titl+'.png', r'_' + titl + '.eps', ), )
    #             fig_types=(r'_' + titl+'.png', ), )

    do_plot(data2, appQ, titles=tuple(str(k+1) + r'in' + str(slice) for k in range(slice)),
            do_show=True, pats=highL, fig_types=('.png', ),
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


if __name__ == "__main__":

    house_number = 5
    slice = 4
    n_app = 3

    do1(house_number, slice)
    # do2(house_number, slice, n_app)
