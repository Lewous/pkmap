# -*- coding: utf-8 -*-
# === ekmapTK.py ===
# ekmapTK using pandas.DataFrame
#   with the help of multiprocess

from numpy import sum
from numpy import fabs, ceil, sqrt
from numpy import log
from numpy import nan, isnan
from numpy import divmod
from numpy import linspace, arange
from numpy import full
from numpy import save, load

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
# import seaborn as sn
from copy import copy
import os

TOTAL_LINE = 6960002
FILE_PATH = './REFIT/CLEAN_House1.csv'
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
        b += (sorted(scope)[half_w], )

    for kn in range(len(a) - width + 1):
        scope = a[kn:kn+width]
        b += (sorted(scope)[half_width], )

    w2 = width
    while w2 > 1:
        w2 -= 2
        half_w = int(0.5 * w2)
        scope = a[0-w2:]
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
    '''
    count how many times each stat appears
    used in multi-processing, so I pack args in this way

    val: the container holds the counting result
        is a dict with `0' defult values
    data1: a fixed DataFrame part to count

    return: counting results
    '''

    val, data1 = arg2
    # `val' is a container
    for k in data1.itertuples():
        # combinate new row as a key of a dict
        nw = ''.join([str(int(u)) for u in k[1:]])

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
        '11100010': 53,        # just an enxmple
        ......
    }
    '''
    # create a dict templet
    # and then fill in the container
    '''
    val0 is the template incase lose of keys()
    '''
    val0 = {}
    nx = int(appQ / 2)
    ny = int(appQ - nx)
    for k in GC(nx):
        for j in GC(ny):
            t = k + j   # is str like '11110001'
            # val0[t] = nan       # for plot benfits
            val0[t] = 0

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
        with Pool() as pool:

            result = pool.map(do_count,  (
                (val0, data0.loc[data0.index.isin(k), c2].copy())
                for k in x2))
            pool.close()
            pool.join()

    toc = time()
    print('finish counting in ' + beauty_time(toc-tic))

    if slice:
        # `slice' is integer, will slice
        # data2 is a list of dict with `slice' items
        data2 = result.copy()

        print(
            f'{sum([sum(list(data2[k].values())) for k in range(slice)])=}' + '\n')
        pass

    else:
        # `slice' is None, won't slice
        # integrate `result' as `data2'
        data2 = val0.copy()

        data2 = {k: sum([result[t][k] for t in range(len(result))])
                 for k in result[0].keys()}

        print(f'{sum(list(data2.values()))=}' + '\n')

    # save data2
    if save_file:
        with open(file_dir + '/EKMap' + file_name[5:] + '.csv', 'w') as f:
            for k in data2.items():
                f.write(':'.join([k[0], str(k[1])]) + '\n')

    return data2


def read_EKfile(file_path):
    """
    loading data from my format
    """
    with open(file_path, 'r') as f:
        data2 = {k.split(':')[0]: int(k.split(':')[1]) for k in f}
    appQ = len(tuple(data2.keys())[0])

    return data2


def new_order(ahead=(), appQ=9):
    """
    ====== inner func ======
    create new order with `ahead' ahead
    ahead: index of app want to ahead in each axes
            so that can be easily obsevered in the Karnaugh Map
            start from 0, used directly as index
            at most two items
            empty item is also accept
    appQ: integer, total number of appliance
        decide the index of second ahead to insert
        also is the length of return tuple

    return: reorderd tuple of index for data2
            like: (4, 0, 1, 2, 7, 3, 5, 6, 8) where (4,7) is ahead

    """

    # wash input argument
    ahead = tuple(int(k) for k in ahead if k > 0 and k < appQ)
    try:
        reod = set(ahead)
    except TypeError as identifier:
        reod = (ahead, )
    nx = int(appQ / 2)      # number of high bits in y-axis
    # ny = int(appQ - nx)     # number of low bits in x-axis

    order2 = list(range(appQ))
    for val, ind in zip(reod, (0, nx, )):
        try:
            order2.remove(val)
            # ensure the item insert inside range
            # avoid over remove when `ahead' have multiple items
            order2.insert(ind, val)
        except ValueError as identifier:
            # `val' out of range, skip
            pass

    return tuple(order2)


def do_plot_single(data3, cmap='inferno', fig_types=(), do_show=True,
                   titles="", pats=[]):
    """
    plot one axe in one figure
    data3: a dict 

    """
    global file_name
    # fill in data
    appQ = len(tuple(data3.keys())[0])  # total number of appliance
    nx = int(appQ / 2)
    ny = int(appQ - nx)

    fig1, ax = plt.subplots(1, 1, figsize=(8, 5))

    # ====== `ekmap' is the contant of a subplot ======
    ekmap = KM(nx, ny)      # preparing a container
    ek = 1
    vmax = log(sum(tuple(data3.values())))

    for _ind in ekmap.index:
        for _col in ekmap.columns:
            d = data3[_ind + _col]
            if d:
                # d > 0
                ekmap.loc[_ind, _col] = log(d)/ek
            # else:
            #     # d == 0
            #     pass
    save('ek0.npy', ekmap)
    # ax.pcolormesh(ekmap, cmap=cmap, vmin=0, vmax=vmax)
    ax.imshow(ekmap, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_yticks(arange(2**nx))
    ax.set_xticks(arange(2**ny))
    ax.set_yticklabels(ekmap.index.values, fontfamily='monospace')
    ax.set_xticklabels(ekmap.columns.values,
                       fontfamily='monospace', rotation=45)
    title = copy(titles)
    if title:
        # `title' has been specified
        ax.set_title(title, size=24)
        # see in https://stackoverflow.com/questions/42406233/
        pass

    if pats:
        # `pats' is not empty, do aditional draw
        for pat in pats:
            ax.add_patch(copy(pat))
            # see in https://stackoverflow.com/questions/47554753
    fig1.tight_layout()

    for fig_type in fig_types:
        plt.pause(1e-13)
        # see in https://stackoverflow.com/questions/62084819/
        plt.savefig('./figs/EKMap' +
                    file_name[5:] + fig_type, bbox_inches='tight')

    if do_show:
        plt.show()
    else:
        plt.close(fig1)

    return 0


def do_plot_multi(data3, cmap='inferno', fig_types=(), do_show=True,
                  titles="", pats=[]):
    """
    plot multiplt axes in one figure

    ====== these have same lenght ======
    data2: a list of EKMap dict
    titles: a list of string

    ====== these shared among axes ======
    pats
    cmap

    """

    global file_name
    # fill in data
    appQ = len(tuple(data3[0].keys())[0])  # total number of appliance
    nx = int(appQ / 2)
    ny = int(appQ - nx)

    # number of slice
    n_slice = len(titles)

    # ====== prepare for canvas distribute ======
    num_row = int(ceil(sqrt(n_slice)))
    num_col = int(ceil(n_slice / num_row))

    # fig1, axes= plt.subplots(num_row, num_col, figsize=(15, 8))
    fsize = (int(num_row * 2**(ny-nx)*3.6), num_col*4)
    fig1, axes = plt.subplots(
        num_col, num_row, figsize=fsize)
    print(f'{fsize=}')

    # ====== `ekmap' is the contant of a subplot ======
    ekmap = KM(nx, ny)      # preparing a container
    ek = 1
    vmax = log(sum(tuple(data3[0].values())))
    for datax, title, ind in zip(data3, titles,
                                 ((c, r) for c in range(num_col) for r in range(num_row))):
        for _ind in ekmap.index:
            for _col in ekmap.columns:
                d = datax[_ind + _col]
                if d:
                    # d > 0
                    ekmap.loc[_ind, _col] = log(d)/ek
                # else:
                #     # d == 0
                #     pass
        ax = axes[ind[0], ind[1]]
        ax.pcolormesh(ekmap, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yticks(arange(2**nx))
        ax.set_xticks(arange(2**ny))
        ax.set_yticklabels(ekmap.index.values, fontfamily='monospace')
        ax.set_xticklabels(ekmap.columns.values,
                           fontfamily='monospace', rotation=45)
        if title:
            # `title' has been specified
            ax.set_title(title, size=24)
            # see in https://stackoverflow.com/questions/42406233/
            pass

        if pats:
            # `pats' is not empty, do aditional draw
            for pat in pats:
                ax.add_patch(copy(pat))
                # see in https://stackoverflow.com/questions/47554753
    fig1.tight_layout()

    for fig_type in fig_types:
        plt.pause(1e-13)
        # see in https://stackoverflow.com/questions/62084819/
        plt.savefig('./figs/EKMap' +
                    file_name[5:] + fig_type, bbox_inches='tight')

    if do_show:
        plt.show()
    else:
        plt.close(fig1)

    return 0


def do_plot(data2, ahead=(), cmap='inferno', fig_types=(), do_show=True,
            titles="", pats=[]):
    """
    do plot, save EKMap figs 

    data2: a dict of EKMap
            or a list of dict of EKMap
    appQ: integer, the number of appliance
    reorder:  put at least two app ahead in each axes
            so that can be easily obsevered in the Karnaugh Map
            start from 0, used directly as index
    fig_types: an enumerable object, tuple here
            used if figure saving required
    do_show: run `plt.show()' or not 
        (fig still showed when set `False', fix later since is harmless)
    title: a string, the fig title 
            must have same size as `data2' (enumerate together)
    pats: add rectangle if needed, an enumerable object

    return: no return

    ====== WARNING ======
    plt.savefig() may occur `FileNotFoundError: [Errno 2]'
    when blending use of slashes and backslashes
    see in https://stackoverflow.com/questions/16333569
    """
    try:
        appQ = len(tuple(data2.keys())[0])  # total number of appliance
    except AttributeError as identifier:
        # type(data2) is a list
        appQ = len(tuple(data2[0].keys())[0])

    order_ind = new_order(ahead, appQ)

    # function reconsitution
    if isinstance(data2, dict):
        # data2 is single
        data3 = {''.join([key[s] for s in order_ind]): data2[key]
                 for key in data2.keys()}
        do_plot_single(data3, cmap, fig_types, do_show,
                       titles, pats)
    else:
        # data2 is a list of dict
        data3 = tuple({''.join([key[s] for s in order_ind]): datax[key]
                       for key in datax.keys()} for datax in data2)
        do_plot_multi(data3, cmap, fig_types, do_show,
                      titles, pats)

    return 0


def slice_REFIT(args):
    """
    slice dataset into `n_slice' pieces
    and save for Mingjun Zhong's code

    file_path: a string
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
    else:
        print("use old `data0'")

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

    # this code will exhibit my novel assessment
    for house_number, slice in ((5, 4, ), ):
        file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'
        data2 = read_REFIT(file_path, slice=slice)

        do_plot(data2, (0,), titles=tuple(str(k+1) + r'in' + str(slice) for k in range(slice)),
                do_show=True, fig_types=('in' + str(slice) + '.png', ),
                )

    t = '='*6
    print(t + ' finished ' + t)
