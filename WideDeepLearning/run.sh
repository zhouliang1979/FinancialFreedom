#!/bin/sh


rm -rf log
mkdir log

rm -rf wdl
mkdir wdl

rm -f nohup.out
nohup python cate_gmv_wdl.py &

exit 0
