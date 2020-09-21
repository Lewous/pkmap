# -*- coding: utf-8 -*- 
# ekmapTK.py
# containing function and variant

# import matplotlib.pyplot as plt

from tqdm import tqdm
# from time import time

TOTAL_LINE = 6960002
FILE_PATH = ""

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


def data_read(filepath = "", ):
    """
    ready data to plot

    filepath: a string, used as open(filepath,'rb')
    return: a dict with keys: unix, agg, app1, app2, ...
    TOTAL_LINE: global variant in EKMApTK

    0. count total lines
    1. read csv file from REFIT
    2. format as each app
    3. median filtrate by app
    4. filtrate to on/off data

    """

    global TOTAL_LINE
    global FILE_PATH
    if filepath == "":
        filepath = FILE_PATH
    TOTAL_LINE = line_count(filepath)

    TOTAL_LINE = 10     # try frist 10 lines for testing
    with tqdm(total = TOTAL_LINE, leave = False, ascii = True, 
            bar_format = "Loading House1...{l_bar}{bar}|{n_fmt}/{total_fmt}") as pybar:
        with open(filepath, 'r') as f:
            f.readline()
            f2 = f.readlines(TOTAL_LINE)
            data0 = [pybar.update(1) or tuple(k.split(',')[1:-1]) for k in f2]
            # warning: data is sting
    # print(data[:1])

    # transfer data as column organized
    with tqdm(leave = False, bar_format = "unzipping data...") as pybar:
        dataz = tuple(zip(*data0))
    
    # del data0
    # application quantity
    appQ = len(dataz) - 1
    # data = {k1:dataz[k1+1] for k1 in range(appQ)}         # is string
    # data = {k1:(int(k2) for k2 in dataz[k1+1]) for k1 in range(appQ)}     # is generator
    with tqdm(total = TOTAL_LINE * appQ, leave = False, ascii = True, 
            bar_format = "Loading data1...{l_bar}{bar}|{n_fmt}/{total_fmt}") as pybar:
        data1 = {k1:tuple((pybar.update(1) or k2)
             for k2 in dataz[k1+1]) for k1 in range(appQ)}
    # data1[0] = dataz[0]
    print("extract data1")

    """
    data1 = {
        # 0: unix     (is string)
        1: app1     (is int)
        2: app2
        ...
        n: appn     (n is appQ)
    }

    """

    data2 = dict()      # filterated data1
    """
    data2 = {
        0: tuple of unix, string
        1: tuple of on/off data of app1, Bool
        2: ...
        ...
        n: n is appQ
        -1: totalline value, integer
    }

    `data2' is a huge dict, which may cause trouble
    (such as memery overflow)
    this will be optimized if necessary
    """

    # clean each app data
    with tqdm(total = TOTAL_LINE * appQ, leave = False, ascii = True, 
            bar_format = "Loading House1...{l_bar}{bar}|{n_fmt}/{total_fmt}") as pybar:
        for kn in range(appQ):
            # kn is 1, 2, 3, ..., appQ
            kn += 1
            # median filter with len of 3
            with tqdm(leave = False, bar_format = "filtrating data1[" + str(kn) + "]...") as pybar2:
                datax = filter(filter(data1[kn], 3), 5)
            # transfer to on/off value
            # boolean may save memory and simplify code
            data2[kn] = tuple([(pybar.update(1) or x > 5) for x in datax])
    data2[0] = dataz[0]
    data2[-1] = TOTAL_LINE

    return data2



if __name__ == "__main__":
    pass
