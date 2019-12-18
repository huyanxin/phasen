#!/usr/bin/env python
# coding=utf-8

import soundfile as sf
import shutil
import os
import sys

def run(src):
    with open('/tmp/t', 'w') as wfid:
        with open(src) as fid:
            for line in fid:
                path = line.strip().split()
                data, fs = sf.read(path[0])
                length = data.shape[0]*1./fs
                ans = '{:s} {:s} {:.4f}\n'.format(path[0], path[1], length)
                wfid.writelines(ans)
    shutil.move('/tmp/t',src)

#run('./tr_wsj0_-5~20.lst')
#run('./cv_wsj0_-5~20.lst')


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print('Error! please run like this:\n'
              '     python add_duration.py ./data/cv_wsj0_-5~20.lst')

        exit(-1)
    run(sys.argv[1])

