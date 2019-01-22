#!/usr/bin/env python 

import sys

cnt = 0
keys = dict()
for line in sys.stdin:
    if cnt == 0:
        for item in line.split(","):
            keys[cnt] = item
            cnt += 1
    else:
        ii = 0
        for item in line.split(","):
            print "{} : {}".format(keys[ii], item)
            ii += 1
