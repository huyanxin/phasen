#!/usr/bin/env python
# coding=utf-8

import os 
import sys
import glob

def get_aeverage(file_name):
    with open(file_name) as fid:
        nums = 0. 
        results = [0]*6 # pesq ori , pesq enh, stoi or, stoi enh, sdr ori sdr enh
        for line in fid:
            tmp = line.strip().split(',')
            nums+=1
            for idx in range(1, len(tmp)):
                results[idx-1] += float(tmp[idx])

        for idx in range(6):
            results[idx]= round(results[idx]/nums, 3)
    return results

if __name__ == '__main__':
   
    if len(sys.argv) == 1:
        print("need exp dir")
        exit(-1)
    tgt = glob.glob(os.path.join(sys.argv[1], '*.csv'))
    sorted_tgt = []
    ref = ['_-5db', '_0db', '_5db', '_10db', '_15db', '_20db']
    for it in ref:
        for tmp in tgt:
            name = tmp.split('/')[-1]
            if it in name:
                sorted_tgt.append(tmp)
                break

    for item in sorted_tgt:
        print(item)
        print(get_aeverage(item))
        print('-'*100)
