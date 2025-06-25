nohup /project/python_env/anaconda3/bin/python TreeModel/tree1.py "1" >tree528_s2964_lr005md9iczs_1.log 2>&1 &
pid1=$!
nohup /project/python_env/anaconda3/bin/python TreeModel/tree1.py "2" >tree528_s2964_lr005md9iczs_2.log 2>&1 &
pid2=$!
nohup /project/python_env/anaconda3/bin/python TreeModel/tree1.py "3" >tree528_s2964_lr005md9iczs_3.log 2>&1 &
pid3=$!
nohup /project/python_env/anaconda3/bin/python TreeModel/tree1.py "4" >tree528_s2964_lr005md9iczs_4.log 2>&1 &
pid4=$!
#wait $pid1 $pid2 $pid3 $pid4