# -*- coding: utf-8 -*-
# === ekmapTK.py ===
# ekmapTK using pandas.DataFrame
#   with the help of multiprocess

from numpy import sum
from numpy import fabs, ceil, sqrt
from numpy import log
from numpy import nan, isnan
from numpy import divmod, mod
from numpy import linspace, arange
from numpy import full
from numpy import save, load
from numpy import array
from numpy import pi
from numpy import meshgrid

from pandas import DataFrame
from pandas import read_csv
from pandas import read_excel
from pandas import concat

from multiprocessing import Pool

from tqdm import tqdm
# import re
import matplotlib.pyplot as plt
import matplotlib.patches as pat
# import seaborn as sn

# python embedded library
from copy import copy
from time import time

from re import findall
from os import listdir
import os
from collections.abc import Iterable 

from appliance_data import appliance_data as app_data

# default global variables
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


def KM(a, b, default=nan):
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

    return DataFrame(full([2**a, 2**b], default),
                     index=GC(a), columns=GC(b))


def GM(n):
    """
    create expond for margin of WKMap
    """

    n = int(abs(n))
    n2 = 2**n
    if n == 0:
        return ()
    elif n == 1:
        return (((1,1.995), ),)
    else:
        # new margin
        new_m = tuple([(2*k+1)/2**(n-1) for k in range(2**(n-1))])
        new_m2 = tuple([(new_m[2*k], new_m[2*k+1]) for k in range(2**(n-2))])
        return tuple([k for k in GM(n-1)] + [new_m2])


def GMI(n):
    """
    create margin index of each four sides
    nx: quantity of variables in x-axis
    ny: quantity of variables in y-axis
    """
    n = int(abs(n))
    if n < 4:
        raise ValueError('n = ' + str(n) + ' < 4 is not accepted!')
    ny = int(n/2)
    nx = n - ny
    # nx, ny = int(abs(nx)), int(abs(ny))
    n_R = int(ny/2)   # marks at right
    n_L = ny - n_R      # marks at left
    n_T = int(nx/2)   # marks at top
    n_B = nx - n_T      # marks at bottom
    n = nx + ny         # total variables

    my = GM(ny)
    tk = True      # pointer, to n_L if True, to n_R if Flase
    nA_L, nA_R = n_L, n_R   # marker of A(nA_L)
    R, L = {}, {}           # output container
    An = 1          # this shares both x and y
    for m in my:
        m2 = tuple([tuple((array(mt)*2**ny-1)/2) for mt in m])
        if tk:
            L['A'+str(An)] = (nA_L, m2)
            tk ^= 1
            An += 1
            nA_L -= 1
        else:
            R['A'+str(An)] = (nA_R, m2)
            tk ^= 1
            An += 1
            nA_R -= 1
    
    mx = GM(nx)
    tk = True      # pointer, to n_L if True, to n_R if Flase
    nA_B, nA_T = n_B, n_T   # marker of A(nA_L)
    B, T = {}, {}           # output container
    for m in mx:
        m2 = tuple([tuple((array(mt)*2**nx-1)/2) for mt in m])
        # print((m, m2))
        if tk:
            B['A'+str(An)] = (nA_B, m2)
            tk ^= 1
            An += 1
            nA_B -= 1
        else:
            T['A'+str(An)] = (nA_T, m2)
            tk ^= 1
            An += 1
            nA_T -= 1
    
    return ((n_L, n_R, n_B, n_T), L, R, B, T)


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


def plot_bar(data2):
    """
    pandas embedded plot.bar will stuck
    
    data2: a pandas series object

    val: return object, a dict of dict
    val['Appliance2'] = {

    }
    """
    # n_data = len(tuple(data2.keys())[0])
    # # is 9 in REFIT

    thrd = int(len(data2)/10000)
    val = {}
    for d in data2.values:
        if d in val.keys():
            val[d] += 1
        else:
            val[d] = 1
    val2 = {}
    for k, it in val.items():
        if it > thrd:
            val2[k] = it
    # plt.bar(val2.keys(), val2.items())
    # plt.show()
    
    return {k:val2[k] for k in sorted(val2.keys())}


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

    # if already have, return directly
    if not slice:
        for dirpath,dirnames,files in os.walk('./'):
            EK_name = 'EKMap' + file_name[5:] + '.txt'
            if EK_name in files:
                with open(file_dir + '/' + EK_name, 'r') as f:
                    data0 = {line.split(':')[0]:int(line.split(':')[1]) for line in f.readlines()}
                print('='*6+' read from '+EK_name+'! '+'='*6)
                return data0
    else:
        for dirpath,dirnames,files in os.walk('./'):
            EK_name = 'EKMap' + file_name[5:] + '_'
            if EK_name+ '1in' + str(slice) +'.txt' in files:
                data2 = [x for x in range(slice)]
                for nslice in range(slice):
                    EK_name2 = EK_name + str(nslice+1) + 'in' + str(slice)
                    with open(file_dir + '/' + EK_name2 + '.txt', 'r') as f:
                        data0 = {line.split(':')[0]:int(line.split(':')[1]) for line in f.readlines()}
                    data2[nslice] = data0
                print('='*6+' read from '+EK_name + 'xin' + str(slice)+'! '+'='*6)
                return data2

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

        print(f'{sum(tuple(data2.values()))=}')

    # save data2
    if save_file and not slice:
        EK_name = 'EKMap' + file_name[5:]
        with open(file_dir + '/' + '.txt', 'w') as f:
            for k in data2.items():
                f.write(':'.join([k[0], str(k[1])]) + '\n')
        print('='*6+' saved '+ EK_name +'! '+'='*6)
    elif save_file:
        EK_name = 'EKMap' + file_name[5:] + '_'
        for nslice, data2 in zip(range(slice), data2):
            EK_name2 = EK_name + str(nslice+1) + 'in' + str(slice)
            with open(file_dir + '/' + EK_name2 + '.txt', 'w') as f:
                for k in data2.items():
                    f.write(':'.join([k[0], str(k[1])]) + '\n')
        print('='*6+' saved '+ EK_name + 'xin' + str(slice) +'! '+'='*6)

    return data2


def read_EKfile(file_path):
    """
    loading data from my format

    not need, imbedded into read_REFIT
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


def do_plot2(data3, cmap='inferno', fig_types=(), do_show=True,
                   titles="", pats=[]):
    """
    plot WKMap with markers around
    using `fig.add_gridspec'
    data3: a dict 

    ====== curtion! the use of x & y are mixed! ======
    """
    global file_name
    print(f'{file_name=}')
    # fill in data
    appQ = len(tuple(data3.keys())[0])  # total number of appliance
    ny = int(appQ / 2)
    nx = appQ - ny

    n_M, *M = GMI(appQ)
    n_L, n_R, n_B, n_T = n_M
    wdd = {         # n_of_variables: (wdx, wdy, magnify_of_lxy)
        0: (0.6, 0.6, 0.4, 0),      # the default parameter
        # 7: (0.8, 1.2),
        4: (1, 1, 0.6, 0), 
        9:(0.4, 0.4, 0.35, -0.2),
    }
    if appQ in wdd.keys():
        wdx, wdy, mg, wfx = wdd[appQ]
    else: 
        wdx, wdy, mg, wfx = wdd[0]
    # wfx: width fixer of the figsize
    # I don't know why there is a slightly offset
    # so a small fixer may be a quick repair pack

    wd1 = 0.3
    wd2 = 0
    lx = mg*2**nx
    ly = mg*2**ny
    # print(f'{((n_L+n_R)*wdx+lx, (n_T+n_B)*wdy+ly)}')
    # print(f'{(n_L*wdx, lx, n_R*wdx)}')
    # print(f'{(n_T*wdy, ly, n_B*wdy)}')

    fig = plt.figure(figsize=((n_L+n_R)*wdx+lx+wfx, (n_T+n_B)*wdy+ly), 
                    constrained_layout=True)
    # fig = plt.figure(figsize=(ly, lx))
    gs = fig.add_gridspec(3, 3,  
                width_ratios=(n_L*wdx, lx, n_R*wdx), 
                height_ratios=(n_T*wdy, ly, n_B*wdy),
                left=0, right=1, bottom=0, top=1,
                wspace=0.02, hspace=0.06)

    ax = fig.add_subplot(gs[1,1])
    # ====== `ekmap' is the contant of a subplot ======
    ekmap = KM(ny, nx)      # preparing a container
    ekback = KM(ny, nx)     # backgroud container
    ek = 1
    vmax = log(sum(list(data3.values())))*0.8
    # vmax = max(tuple(data3.values()))
    # vmin = min(tuple(data3.values()))
    # vmax = 110
    vmin = 0

    for _ind in ekmap.index:
        for _col in ekmap.columns:
            d = data3[_ind + _col]
            if d:
                # d is larger than 0
                ekmap.loc[_ind, _col] = log(d/ek)
                # ekmap.loc[_ind, _col] = d
            else:
                # d equals 0
                ekback.loc[_ind, _col] = 0.06
    
    # print(f'{(vmax, vmin)=}')
    ax.imshow(ekback, cmap='Blues',vmin=0, vmax=1)
    ax.imshow(ekmap, alpha = 1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(arange(2**ny))
    ax.set_xticks(arange(2**nx))
    ax.set_yticklabels(ekmap.index.values, fontfamily='monospace')
    ax.set_xticklabels(ekmap.columns.values,
                       fontfamily='monospace', rotation=45)
    if pats:
        # `pats' is not empty, do aditional draw
        for pat in pats:
            ax.add_patch(copy(pat))
    ax.axis('off')

    for S, pl, xy in zip(M, (gs[1, 0], gs[1,2], gs[2,1], gs[0,1]), ('L','R','B','T')):
        # S = L, R, B, T
        if xy in ('B', 'T'):
            ax_S = fig.add_subplot(pl, sharex=ax)
        elif xy in ('L', 'R'):
            ax_S = fig.add_subplot(pl, sharey=ax)
        else:
            ax_S = fig.add_subplot(pl)
        # print(f'{S}')
        cf1 = wd1
        cf2 = 0.1
        for margin in S.items():
            mag_n = margin[0]
            # dict.key, like `A1'
            mag_v = margin[1]
            # dict.value like `(2, ((3.5, 7.48),))'
            # or `(1, ((0.5, 2.5), (4.5, 6.5)))'
            ofst = mag_v[0]     # offset, a number start from 1
            args_text = {'ha':'center', 'va':'center', 'backgroundcolor':'w', 
                        'family':'monospace', 'size':'xx-large'}
            for ind in mag_v[1]:
                # `mag_v[1]' like `((3.5, 7.48),)'
                # or `((0.5, 2.5), (4.5, 6.5))'
                m_st = ind[0]           # start
                m_itl = ind[1] - ind[0]     # interval
                m_tx = m_st + m_itl/2       # coordinate of text, parallel to the axis
                m_ty = cf1*ofst - cf2       # vartical to the axis
                if   xy in ('L', ):
                    ax_S.add_patch(plt.Rectangle(
                        (cf1, m_st), 0-cf1-m_ty, m_itl, fill=False))
                    ax_S.text(0-m_ty, m_tx, mag_n, **args_text)
                elif xy in ('R', ):
                    ax_S.add_patch(plt.Rectangle(
                        (0-cf1, m_st), cf1+m_ty, m_itl, fill=False))
                    ax_S.text(m_ty, m_tx, mag_n, **args_text)
                elif xy in ('B', ):
                    ax_S.add_patch(plt.Rectangle(
                        (m_st, cf1), m_itl, 0-cf1-m_ty, fill=False))
                    ax_S.text(m_tx, 0-m_ty, mag_n, **args_text)
                elif xy in ('T', ):
                    ax_S.add_patch(plt.Rectangle(
                        (m_st, 0-cf1), m_itl, cf1+m_ty, fill=False))
                    ax_S.text(m_tx, m_ty, mag_n, **args_text)

        if   xy in ('L', ):
            ax_S.set_xlim(left=0-n_L*wd1+wd2, right=0)
        elif xy in ('R', ):
            ax_S.set_xlim(left=0, right=n_R*wd1-wd2)
        elif xy in ('B', ):
            ax_S.set_ylim(bottom=0-n_B*wd1+wd2, top=0)
        elif xy in ('T',):
            ax_S.set_ylim(bottom=0, top=n_T*wd1-wd2)
            if titles:
                ax_S.set_title(titles, size=24)
        ax_S.axis('off')

    # fig.tight_layout()
    # if not isinstance(fig_types, Iterable):
    #     fig_types = (fig_types, )
    for fig_type in fig_types:
        plt.pause(1e-13)
        # see in https://stackoverflow.com/questions/62084819/
        plt.savefig('./figs/EKMap' +
                    file_name[5:] + 
                    # str(time())[-5:] + 
                    fig_type, 
                    bbox_inches='tight')

    if do_show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def do_plot3(data3, cmap='inferno', fig_types=(), do_show=True,
                   titles="", pats=[]):
    """
    plot WKMap with 3D surface
    data3: a dict 

    """
    global file_name
    print(f'{file_name=}')
    # fill in data
    appQ = len(tuple(data3.keys())[0])  # total number of appliance
    ny = int(appQ / 2)
    nx = appQ - ny

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    # ====== `ekmap' is the contant of a subplot ======
    ekmap = KM(ny, nx, 0)      # preparing a container
    ek = 1
    for _ind in ekmap.index:
        for _col in ekmap.columns:
            d = data3[_ind + _col]
            if d:
                # d is greater than 0
                ekmap.loc[_ind, _col] = log(d)/ek
            else:
                # d equals 0
                pass

    X = arange(2**nx)
    Y = arange(2**ny)
    X, Y = meshgrid(X, Y)
    ax.plot_surface(X, Y, ekmap, cmap=cmap)
    ax.set_yticks(arange(2**ny))
    ax.set_xticks(arange(2**nx))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylabel(r'$A_1A_2A_3A_4$', 
                family='monospace', size=16)
    ax.set_xlabel(r'$A_5A_6A_7A_8A_9$',
                family='monospace', size=16)
    # if pats:
    #     # `pats' is not empty, do aditional draw
    #     for pat in pats:
    #         ax.add_patch(copy(pat))
    ax.set_zticks([])
    # ax.axis('off')

    # fig.tight_layout()
    if not isinstance(fig_types, Iterable):
        fig_types = (fig_types, )
    for fig_type in fig_types:
        plt.pause(1e-13)
        # see in https://stackoverflow.com/questions/62084819/
        plt.savefig('./figs/EKMap' +
                    file_name[5:] + 
                    # str(time())[-5:] + 
                    fig_type, 
                    bbox_inches='tight')

    if do_show:
        plt.show()
    else:
        plt.close(fig)

    return fig


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

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # ====== `ekmap' is the contant of a subplot ======
    ekmap = KM(nx, ny)      # preparing a container
    ekback = KM(nx, ny)     # backgroud color
    ek = 1
    vmax = log(sum(tuple(data3.values())))

    for _ind in ekmap.index:
        for _col in ekmap.columns:
            d = data3[_ind + _col]
            if d:
                # d is greater than 0
                ekmap.loc[_ind, _col] = log(d)/ek
            else:
            #     # d equals 0
                ekback.loc[_ind, _col] = 0.02
    # save('ek0.npy', ekmap)
    # ax.pcolormesh(ekmap, cmap=cmap, vmin=0, vmax=vmax)

    ax.imshow(ekback, cmap='Blues',vmin=0, vmax=1)
    ax.imshow(ekmap, alpha = 1, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_yticks(arange(2**nx))
    ax.set_xticks(arange(2**ny))
    ax.set_yticklabels(ekmap.index.values, 
        family='monospace', size='xx-large')
    ax.set_xticklabels(ekmap.columns.values,
        fontfamily='monospace', size='xx-large', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    xlabel = "".join([r'$A_' + str(x+1) + r'$' for x in range(ny)])
    ylabel = "".join([r'$A_' + str(x+1) + r'$' for x in range(ny, ny+nx)])
    ax.set_ylabel(xlabel,family='monospace', size=20)
    ax.set_xlabel(ylabel,family='monospace', size=20)
    

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
    fig.tight_layout()

    if not isinstance(fig_types, Iterable):
        fig_types = (fig_types, )
    for fig_type in fig_types:
        plt.pause(1e-13)
        # see in https://stackoverflow.com/questions/62084819/
        plt.savefig('./figs/EKMap' +
                    file_name[5:] +str(time())[-5:] + fig_type, 
                    bbox_inches='tight')

    if do_show:
        plt.show()
    else:
        plt.close(fig)

    return fig


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
    n_slice = len(data3)

    # ====== prepare for canvas distribute ======
    num_row = int(ceil(sqrt(n_slice)))
    num_col = int(ceil(n_slice / num_row))

    fsize = (int(num_row * 2**(ny-nx)*3), num_col*4)
    fig, axes = plt.subplots(
        num_col, num_row, figsize=fsize)
    print(f'{fsize=}')

    # ====== `ekmap' is the contant of a subplot ======
    ekmap = KM(nx, ny)      # preparing a container
    ekback = KM(nx, ny)
    ek = 1
    vmax = log(sum(tuple(data3[0].values())))
    # ax_it = (k for k in axes)
    if not titles:
        titles = full(n_slice, None)
    ind = ((c, r) for c in range(num_col) for r in range(num_row))
    for datax, title, in zip(data3, titles, ):
        for _ind in ekmap.index:
            for _col in ekmap.columns:
                d = datax[_ind + _col]
                if d:
                    # d is greater than 0
                    ekmap.loc[_ind, _col] = log(d)/ek
                else:
                    # d equals 0
                    ekback.loc[_ind, _col] = 0.02
        ind2 = next(ind)
        ax = axes[ind2[0], ind2[1]]
        # ax.pcolormesh(ekmap, cmap=cmap, vmin=0, vmax=vmax)
        ax.imshow(ekback, cmap='Blues', vmin=0, vmax=1)
        ax.imshow(ekmap, cmap=cmap, vmin=0, vmax=vmax)

        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_yticks(arange(2**nx))
        # ax.set_xticks(arange(2**ny))
        # ax.set_yticklabels(ekmap.index.values, fontfamily='monospace')
        # ax.set_xticklabels(ekmap.columns.values,
        #                 fontfamily='monospace', rotation=45)
        ax.axis('off')
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
    # clean the rest axes
    for ind2 in ind:
        ax = axes[ind2[0], ind2[1]]
        ax.axis('off')

    fig.tight_layout()

    for fig_type in fig_types:
        plt.pause(1e-13)
        # see in https://stackoverflow.com/questions/62084819/
        plt.savefig('./figs/EKMap' +
                    file_name[5:] + fig_type, bbox_inches='tight')

    if do_show:
        plt.show()
    else:
        plt.close(fig)

    return fig


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
            ## string along is not excepted ## 
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

    # input correction
    if isinstance(fig_types, str):
        fig_types = (fig_types, )

    # function reconsitution
    if isinstance(data2, dict):
        # data2 is single
        data3 = {''.join([key[s] for s in order_ind]): data2[key]
                 for key in data2.keys()}
        print('do_plot_single')
        do_plot2(data3, cmap, fig_types, do_show,
                       titles, pats)
    else:
        # data2 is a list of dict
        data3 = tuple({''.join([key[s] for s in order_ind]): datax[key]
                       for key in datax.keys()} for datax in data2)
        print('do_plot_multiple')
        do_plot_multi(data3, cmap, fig_types, do_show,
                      titles, pats)

    return 0


def slice_REFIT(file_path, n_slice, n_valid, n_test, n_app, save_dir):
    """
    slice dataset into `n_slice' pieces
    and save for Mingjun Zhong's code

    file_path: a string
    # house_No: integer, index of house, start from 1 
    n_slice: integer, pieces of slicing
    n_valid: integer, number of slice for validation
    n_test: integer, number of slice for testing
    n_app: integer, number of appliance, start from 1

    can using with: fun(**args_dict)
    """
    global data0
    file_name = findall(r'/(.+)\.csv', file_path)[-1]
    file_dir = '/'.join(file_path.split('/')[:-1])
    house_No = findall(r'\d+', file_name)[-1]
    # print(f'{house_No=}')

    if data0.empty:
        # reuse `data0' when slicing multiple times
        with tqdm(leave=False,
                  bar_format="reading " + file_name + " ...") as pybar:
            data0 = read_csv(file_path)
    else:
        print("use old `data0'")

    app = 'Appliance'+str(n_app)    # such as 'Appliance3'

    # get name_app
    name_app = read_excel(os.path.join(file_dir, 'MetaData_Tables.xlsx'), 
                sheet_name='House '+str(house_No), 
                usecols=('Aggregate', ), ).values[n_app-1][0]
    # name_app = 'freezer'
    name_app = name_app.replace(' ', '_')

    if not save_dir:
        save_dir = './_exam_'
    save_dir = os.path.join(save_dir, name_app)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('create ' + f'{save_dir}')
    else:
        # removeall files in `save_dir`
        # have a clean working dir
        for fdir, links, files in os.walk(save_dir):
            [os.remove(os.path.join(fdir, f2)) for f2 in files]
        print('cleaned ' + f'{save_dir}')

    # mean_agg = data0['Aggregate'].mean()
    # std_agg = data0['Aggregate'].std()
    mean_agg = 566
    std_agg = 843
    # datax = data0[app]
    # mean_app = datax[(datax>5) & (datax < 800)].mean()
    # std_app = datax[(datax>5) & (datax < 800)].std()
    mean_app = 50
    std_app = 13
    mean_app = app_data[name_app]['mean']
    std_app = app_data[name_app]['std']
    TOTAL_LINE = len(data0.index)
    print(f'{TOTAL_LINE=}')
    print(f'{(mean_agg, std_agg)=}')
    print(f'{(mean_app, std_app)=}')

    x1 = linspace(0, TOTAL_LINE, num=n_slice + 1, dtype='int')
    # x1 is a list
    print(f'{x1=}')
    x2 = ((x1[k], x1[k+1]) for k in range(n_slice))
    if not isinstance(n_test, Iterable):
        n_test = (n_test, )
    if not isinstance(n_valid, Iterable):
        n_valid = (n_valid, )
    
    throd=8
    for ind, k in enumerate(x2):
        ind += 1
        print(f'{(ind, k)}')
        datax = data0.loc[k[0]:k[1], ['Aggregate', app]]
        data_agg = (datax['Aggregate'] - mean_agg) / std_agg
        data_app = (datax[app] - mean_app) / std_app
        data2save = concat([data_agg, data_app], axis=1)
        if ind in n_test:
            # is test set
            data2save.to_csv('/'.join([save_dir, name_app]) 
                             + '_test_' + 'S' + str(n_test)
                             + '.csv', index=False, mode='a', header=False)
            dataw = data0.loc[k[0]:k[1], 'Appliance1':'Appliance9']
            with open('/'.join([save_dir, name_app])
                             + '_test_codings.txt', 'a') as f:
                for l in dataw.itertuples(index=False):
                    f.write(''.join([str(int(u>throd)) for u in l]))
                    f.write('\n')
            print('\tslice ' + str(ind) + ' for testing')
        elif ind in n_valid:
            # is validation set
            data2save.to_csv('/'.join([save_dir, name_app]) 
                             + '_valid_' + 'S' + str(n_valid)
                             + '.csv', index=False, mode='a', header=False)
            dataw = data0.loc[k[0]:k[1], 'Appliance1':'Appliance9']
            with open('/'.join([save_dir, name_app])
                             + '_valid_codings.txt', 'a') as f:
                for l in dataw.itertuples(index=False):
                    f.write(''.join([str(int(u>throd)) for u in l]))
                    f.write('\n')
            print('\tslice ' + str(ind) + ' for validation')
        else:
            # is training set
            data2save.to_csv('/'.join([save_dir, name_app]) 
                             + '_training_'
                             + '.csv', index=False, mode='a', header=False)
            dataw = data0.loc[k[0]:k[1], 'Appliance1':'Appliance9']
            with open('/'.join([save_dir, name_app])
                             + '_train_codings.txt', 'a') as f:
                for l in dataw.itertuples(index=False):
                    f.write(''.join([str(int(u>throd)) for u in l]))
                    f.write('\n')
    return None


if __name__ == "__main__":
    files = findall('(CLEAN_House[0-9]+).csv', '='.join(listdir('REFIT')))
    # files = ['CLEAN_House8']
    # print(f'{files=}')

    # this code will exhibit my novel assessment
    # for house_number, slice in ((5, 4, ), ):
    #     file_path = 'REFIT/CLEAN_House' + str(house_number) + '.csv'
    #     data2 = read_REFIT(file_path, slice=slice)

    #     do_plot(data2, (0,), titles=tuple(str(k+1) + r'in' + str(slice) for k in range(slice)),
    #             do_show=True, fig_types=('in' + str(slice) + '.png', ),
    #             )
    x = GMI(7)
    for ind in x:
        print((ind))
    t = '='*6
    print(t + ' finished ' + t)
