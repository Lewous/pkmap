 # -*- coding: utf-8 -*- 
# ekmapTK.py
# ekmapTK using pandas
# keep original ekmapTK as ekmapTK0

from numpy import fabs
from numpy import log
from numpy import nan

from pandas import DataFrame
from pandas import read_csv

from tqdm import tqdm
# from time import time

# TOTAL_LINE = 6960002
FILE_PATH = 'REFIT/CLEAN_House1.csv'


def line_count(filepath):
    """
    to count the total lines in a file to read

    filepath: a string, used as open(filepath,'rb')
    return: a integer if success
    TOTAL_LINE: global variant in EKMApTK

    warning: no input filter, might caught bug 
        if called roughtly.
    """

    # global TOTAL_LINE
    global FILE_PATH
    FILE_PATH = filepath
    with open(filepath,'rb') as f:
        count = 0
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
    # TOTAL_LINE = count
    print("find " + str(count-1) + " lines data")
    return count


def filter(a, width = 3):
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


def data_read(filepath = "", ):
    """
    ready data to plot

    filepath: a string, used as open(filepath,'rb')
    return: a dict with keys: unix, agg, app1, app2, ...
    # TOTAL_LINE: global variant in EKMApTK

    0. count total lines
    1. read csv file from REFIT
    2. format as each app
    3. median filtrate by app
    4. filtrate to on/off data

    """
    threshold = 5       # power data large than this view as on state

    # global TOTAL_LINE
    global FILE_PATH
    if filepath == "":
        filepath = FILE_PATH
    filename = filepath.split('/')[-1].split('.')[:-1][0]

    with tqdm(leave = False, bar_format = "reading " + filename + " ...") as pybar:
        data0 = read_csv(filepath)

    # appliance quantity
    appQ = len(data0.columns) - 4
    print("find `" + str(appQ) + "' appliance with `" + 
            str(len(data0.index)) + "' lines")

    data0.rename(columns = {'Appliance' + str(k+1): 'app' + str(k+1) for k in range(appQ)}) 
    # data0.columns is [Time  Unix  Aggregate  app1  app2  app3  app4  app5  app6  app7  app8  app9  Issues]

    # filter here 
    # add later as not necessary
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
    val0 = {}
    for k in GC(4):
        for j in GC(4):
            t = k + j   # is str like '11110001'
            val0[t] = nan
            print(t+': ' + str(val0[t]))
    # print([k + ': ' + str(val0[k]) for k in val0.keys()])
    


    
