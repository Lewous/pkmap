# -*- coding: utf-8 -*- 

# test and vertify file

# from ekmapTK import line_count as lc
# from ekmapTK import filter
from ekmapTK import data_read as dr

# from tqdm import tqdm

wt = 50      # width per part taken
itl = 1/wt   # interval  is 1/50


# import dataset, REFIT in this case
# and reformat
f = 'REFIT/CLEAN_House1.csv'
data2 = dr(f)
# total_line = data2[-1]
# print(total_line)

t = '='*6
print(t + 'done' + t)