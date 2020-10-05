# -*- coding: utf-8 -*-
# ekmapTK.py
# ekmapTK using pandas.DataFrame
#   with the help of multiprocess
# keep original ekmapTK as ekmapTK0

# from pandas.core.indexes.datetimes import date_range
# from ekmapTK0 import TOTAL_LINE

import re
from numpy import sum
from numpy import fabs
from numpy import log
from numpy import nan, isnan
from numpy import divmod
from numpy import linspace
from numpy import full
from numpy.core.numeric import outer

from pandas import DataFrame
from pandas import read_csv
from pandas import concat
from multiprocessing import Pool

from time import time
from tqdm import tqdm
# import re
from re import findall
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import seaborn as sn
from copy import copy
import os

TOTAL_LINE = 6960002
FILE_PATH = 'REFIT/CLEAN_House1.csv'
val0 = {}
data0 = DataFrame([])
file_name = 'Housex'


def line_count(file_path):
    """
    to count the total lines in a file to read

    file_path: a string, used as open(file_path,'rb')
    return: a integer if success
    TOTAL_LINE: global variant in EKMApTK

    warning: no input filter, might caught bug 
        if called roughtly.
    """

    # global TOTAL_LINE
    global FILE_PATH
    FILE_PATH = file_path
    with open(file_path, 'rb') as f:
        count = 0
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
    # TOTAL_LINE = count
    print("find " + str(count-1) + " lines data")
    return count


def filter(a, width=3):
    """
    median filtrate executor

    a: object to filter, an one dimensional list or tuple
        need run len(a)
    width: filter width, like 3, 5, ...
    # orig: return original value if is True
    #     return on/off value if is False

    return: an a-size tuple created with same length as `a'
    """

    half_width = int(0.5 * width)
    b = (a[0], )

    w2 = 1
    while w2 < width - 2:
        w2 += 2
        half_w = int(0.5 * w2)
        scope = a[:w2]
        # print('w2 is ' + str(w2))
        # print(scope)
        b += (sorted(scope)[half_w], )

    for kn in range(len(a) - width + 1):
        scope = a[kn:kn+width]
        # print(scope)
        b += (sorted(scope)[half_width], )

    w2 = width
    while w2 > 1:
        w2 -= 2
        half_w = int(0.5 * w2)
        scope = a[0-w2:]
        # print('w2 is ' + str(w2))
        # print(scope)
        b += (sorted(scope)[half_w], )
    return b


def GC(n):
    """
    generate `Gray Code' using self-calling function

    n: an integer greater than zero (n>=0)
    return: tuple (string type of binary code)

    """

    # Gray Code generator
    n = int(fabs(n))
    if n == 1:
        return ('0', '1')
    elif n == 0:
        return ()
    else:
        a = GC(n-1)
        return tuple(['0'+k for k in a] + ['1'+k for k in a[::-1]])


def KM(a, b):
    """
    generate Karnaugh map template
    ====== template only ======
    default value is np.nan for plotting benfits

    a: an integer, number of variable in row
    b: an integer, number of variable in col
    return: a pd.DataFrame with GC(a) * GC(b)
    """

    a = int(fabs(a))
    b = int(fabs(b))

    return DataFrame(full([2**a, 2**b], nan),
                     index=GC(a), columns=GC(b))


def beauty_time(time):
    """
    beauty time string
    time: time in seconds
    return: time in string

    warning: using a feature published in python 3.8
    """
    d = 0
    h = 0
    m = 0
    s = 0
    ms = 0
    str_time = ""
    if time > 3600 * 24:
        (d, time) = divmod(time, 3600*24)
        str_time += f"{int(d)}d "
    if time > 3600:
        (h, time) = divmod(time, 3600)
        str_time += f"{int(h)}h "
    if time > 60:
        (m, time) = divmod(time, 60)
        str_time += f"{int(m)}m "
    (s, ms) = divmod(time*1000, 1000)
    str_time += f"{int(s)}s {int(ms)}ms"

    return str_time


def do_count(arg2):
    val, data1 = arg2
    # print(x2)
    for k in data1.itertuples():
        # combinate new row as a key of a dict

        # used in .iterrows()
        # nw = ''.join(k[1].astype('int8').astype('str'))
        # used in .itertuples()
        nw = ''.join([str(int(k)) for k in k[1:]])
        # nw = ''.join([str(k) for k in int8(k[1:])])

        # for nan default
        # if isnan(val[nw]):
        #     val[nw] = 1
        # else:
        #     val[nw] += 1

        # for 0 default
        val[nw] += 1

    return val


def read_REFIT(file_path="", save_file=False, slice=None):
    """
    ready data to plot

    file_path: a string, used as open(file_path,'rb')
    save_file: save EKMap data or not
    slice: slice or not
        is None: no slice
        is integer: slice dataset into `slice' piece
        == this will affect the process number `PN' of multiprocess

    return: 
        data2: a dict of EKMap:
            '01010000': 35,     # counting consequence
            '01011000': 0,      # not appear
            ......
        appQ: number of total appliance

    # TOTAL_LINE: global variant in EKMApTK

    0. count total lines
    1. read csv file from REFIT
    2. format as each app
    3. median filtrate by app
    4. filtrate to on/off data

    """
    threshold = 5       # power data large than this view as on state

    global TOTAL_LINE
    global FILE_PATH
    global file_name
    # global data0

    if file_path == "":
        file_path = FILE_PATH
    file_name = findall('/(.+)\.', file_path)[0]
    file_dir = '/'.join(file_path.split('/')[:-1])
    # file_path.split('/')[-1].split('.')[:-1][0]

    # if data0.empty:
    with tqdm(leave=False,
              bar_format="reading " + file_name + " ...") as pybar:
        data0 = read_csv(file_path)

    TOTAL_LINE = len(data0.index)
    # appliance total number
    appQ = len(data0.columns) - 4
    print("find `" + str(appQ) + "' appliance with `" +
          str(TOTAL_LINE) + "' lines data in " + file_name)

    # data0.rename(columns = {'Appliance' + str(k+1): 'app' + str(k+1)
    #     for k in range(appQ)})
    '''
    data0.columns is: 
      ['Time', 'Unix', 'Aggregate', 'Appliance1', 'Appliance2', 'Appliance3',
       'Appliance4', 'Appliance5', 'Appliance6', 'Appliance7', 'Appliance8',
       'Appliance9', 'Issues']
    '''

    '''
    # filter here 
    # add later as it's not necessary

    '''

    # transfer to on/off value
    dx = data0.loc[:, 'Appliance1': 'Appliance9']
    data0.loc[:, 'Appliance1': 'Appliance9'] = (dx > threshold)

    '''
    counting
    store statics in a dict:
    val0 = {
        '11100010': 53,        # just an example
        ......
    }
    '''
    # create a dict templet
    # and then fill in the container
    '''
    val0 is the template incase lose of keys()
    '''
    val0 = {}
    xa = int(appQ / 2)
    xb = int(appQ - xa)
    for k in GC(xa):
        for j in GC(xb):
            t = k + j   # is str like '11110001'
            # val0[t] = nan       # for plot benfits
            val0[t] = 0
            # print(t+': ' + str(val0[t]))
    # print([k + ': ' + str(val0[k]) for k in val0.keys()])

    # fill in statics
    # c2: choose 8 app to analysis (app3 don't looks good)
    c2 = findall('Appliance[0-9]+', ''.join(data0.columns))
    # c2 is a list of string
    tic = time()
    # PN means number of process
    if slice:
        # if `slice' is integer, do slice, offer PN as `slice'
        PN = slice
        pass
    else:
        # if `slice` is None, no slice, offer PN as 8
        PN = 8

    x1 = linspace(0, TOTAL_LINE/1, num=PN + 1, dtype='int')
    # x1 is a list of
    x2 = (range(x1[k], x1[k+1]) for k in range(PN))
    # x2 is a generator of each scope in a tuple of two int
    print(x1)
    # result = list(range(PN))
    with tqdm(leave=False, bar_format="Counting ...") as pybar:
        # with tqdm(total = TOTAL_LINE * appQ, leave = False, ascii = True,
        #         bar_format = "Counting ...{l_bar}{bar}|{n_fmt}/{total_fmt}") as pybar:
        with Pool() as pool:
            # for k in range(PN):
            #     x = next(x2)
            #     print(x)
            result = pool.map(do_count,  (
                (val0.copy(), data0.loc[data0.index.isin(k), c2].copy())
                for k in x2))
            pool.close()
            pool.join()

    toc = time()
    print('finish counting in ' + beauty_time(toc-tic))

    if slice:
        # `slice' is integer, will do slice
        # data2 is a list of dict with `slice' items
        data2 = result.copy()

        print(
            f'{sum([sum(list(data2[k].values())) for k in range(slice)])=}' + '\n')
        pass

    else:
        # `slice' is None, no slice
        data2 = val0.copy()
        # for k in result[0].keys():
        #     x0 = (result[t][k] for t in range(len(result)))
        #     # include np.nan
        #     x1 = sum([0 if isnan(t) else t for t in x0], dtype='int')
        #     data2[k] = x1
        data2 = {k: sum([result[t][k] for t in range(len(result))])
                 for k in result[0].keys()}
        # print(data2.items())

        # [print(k) for k in data2.items()]
        print(f'{sum(list(data2.values()))=}' + '\n')
        # print(sum(data2.values()))

    # save data2
    if save_file:
        with open(file_dir + '/EKMap' + file_name[5:] + '.csv', 'w') as f:
            for k in data2.items():
                f.write(':'.join([k[0], str(k[1])]) + '\n')

    return data2, appQ


def read_EKfile(file_path):
    """
    loading data from my format
    """
    with open(file_path, 'r') as f:
        data2 = {k.split(':')[0]: int(k.split(':')[1]) for k in f}
    appQ = len(tuple(data2.keys())[0])

    return data2, appQ


def do_plot(data2, appQ, cmap='inferno', fig_type=(), do_show=True,
            title="", pats=[]):
    """
    do plot, save EKMap figs 

    data2: a dict of EKMap
    appQ: integer, the number of appliance
    cmap: 
    fig_type: an enumerable object, tuple here
    do_show: run `plt.show()' or not 
        (fig still showed, fix later since is harmless)
    title: a string, the fig title 

    return: no return

    ====== WARNING ======
    plt.savefig() may occur `FileNotFoundError: [Errno 2]'
    when blending use of slashes and backslashes
    see in https://stackoverflow.com/questions/16333569
    """

    global file_name
    # fill in data
    xa = int(appQ / 2)
    xb = int(appQ - xa)
    ekmap = KM(xa, xb)
    # ek = log(data2['0' * appQ])
    # ek = log(max(data2.values()))
    # ek = log(appQ)
    ek = 1

    for _ind in ekmap.index:
        for _col in ekmap.columns:
            d = data2[_ind + _col]
            if d:
                # d > 0
                ekmap.loc[_ind, _col] = log(d)/ek
            else:
                # d == 0
                pass

    # print(ekmap)
    sn.set()
    plt.figure(figsize=(15, 8))
    # cmap = 'inferno'
    ax = sn.heatmap(ekmap, cbar=False, cmap=cmap)
    plt.ylabel('High ' + str(xa) + ' bits', size=18)
    plt.xlabel('Low ' + str(xb) + ' bits', size=18)
    plt.yticks(rotation='horizontal')
    plt.xticks(rotation=45)
    if title:
        # `title' has been specified
        ax.set_title(title, size=24)
        # see in https://stackoverflow.com/questions/42406233/
        pass

    if pats:
        # pats is not empty
        for k in pats:
            # newk = copy(k)
            ax.add_patch(copy(k))
            # see in https://stackoverflow.com/questions/47554753

    for fig_type in fig_type:
        plt.pause(1e-13)
        # see in https://stackoverflow.com/questions/62084819/
        plt.savefig('REFIT/EKMap' +
                    file_name[5:] + fig_type, bbox_inches='tight')

    if do_show:
        plt.show()

    return 0


def slice_REFIT(args):
    """
    slice dataset into `n_slice' pieces
    and save for Mingjun Zhong's code

    file_path: 
    n_slice: integer, number to slice, 10 for house7
    n_valid: integer, number of slice for validation
    n_test: integer, number of slice for testing
    n_app: integer, number of appliance for analysising

    """
    global data0
    file_path, n_slice, n_valid, n_test, n_app, save_dir = args
    file_name = findall('/(.+)\.', file_path)[0]
    file_dir = '/'.join(file_path.split('/')[:-1])
    # file_path.split('/')[-1].split('.')[:-1][0]

    if data0.empty:
        # reuse `data0' when slicing multiple times
        with tqdm(leave=False,
                  bar_format="reading " + file_name + " ...") as pybar:
            data0 = read_csv(file_path)

    app = 'Appliance'+str(n_app)    # such as 'Appliance3'
    name_app = 'freezer'
    if not save_dir:
        save_dir = './' + name_app + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # mean_agg = data0['Aggregate'].mean()
    # std_agg = data0['Aggregate'].std()
    mean_agg = 566
    std_agg = 843
    # datax = data0[app]
    # mean_app = datax[(datax>5) & (datax < 800)].mean()
    # std_app = datax[(datax>5) & (datax < 800)].std()
    mean_app = 50
    std_app = 13
    TOTAL_LINE = len(data0.index)
    print(f'{TOTAL_LINE=}')
    print(f'{(mean_agg, std_agg)=}')
    print(f'{(mean_app, std_app)=}')

    x1 = linspace(0, TOTAL_LINE, num=n_slice + 1, dtype='int')
    # x1 is a list
    print(f'{x1=}')
    x2 = ((x1[k], x1[k+1]) for k in range(n_slice))
    for ind, k in enumerate(x2):
        ind += 1
        print(f'{(ind, k)=}')
        datax = data0.loc[k[0]:k[1], ['Aggregate', app]]
        data_agg = (datax['Aggregate'] - mean_agg) / std_agg
        data_app = (datax[app] - mean_app) / std_app
        data2save = concat([data_agg, data_app], axis=1)
        if ind == n_test:
            # is test set
            data2save.to_csv(save_dir + name_app + '_test_' + 'S' + str(n_test)
                             + '.csv', index=False)
            print('\tslice ' + str(ind) + ' for testing')
        elif ind == n_valid:
            # is validation set
            data2save.to_csv(save_dir + name_app + '_valid_' + 'S' + str(n_valid)
                             + '.csv', index=False)
            print('\tslice ' + str(ind) + ' for validation')
        else:
            # is training set
            data2save.to_csv(save_dir + name_app + '_training_'
                             + '.csv', index=False, mode='a', header=False)

    return None


if __name__ == "__main__":
    files = findall('(CLEAN_House[0-9]+).csv', '='.join(listdir('REFIT')))
    # files = ['CLEAN_House8']
    print(f'{files=}')

    t = '='*6
    print(t + ' finished ' + t)
