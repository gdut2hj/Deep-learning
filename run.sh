#! /bin/bash
#python3 -u "$@" > ./logs/log.txt 2>&1 &
for key in "$@"
do 
    python3 -u $key > ./logs/log.txt 2>&1 & 
done
