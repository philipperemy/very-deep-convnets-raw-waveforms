#!/usr/bin/env bash
rm nohup.out
rm out.tsv
nohup python3 -u model_run.py &
sleep 2
tail -f nohup.out
