# -*- coding: utf-8 -*- 
# ekmapTK.py
# ekmapTK using pandas
# keep original ekmapTK as ekmapTK0

from pandas import DataFrame
from pandas import read_csv

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
    filename = filepath.split('/')[-1]
    
    TOTAL_LINE = line_count(filepath)

    TOTAL_LINE = 10     # try frist 10 lines for testing
    with tqdm(leave = False, bar_format = "reading " + filename + " ...") as pybar:
        data0 = read_csv(filepath)



