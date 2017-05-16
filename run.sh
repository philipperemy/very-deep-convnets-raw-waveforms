rm nohup.out
rm out.tsv
nohup python3 -u model_run_all.py &
sleep 2
tail -f nohup.out
