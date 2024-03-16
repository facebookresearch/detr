echo $(date)
conda config --append envs_dirs /srip-vol/parth/myenvs
cd /srip-vol/yuze/detr/
python=/srip-vol/parth/myenvs/detr/bin/python

$python main.py

echo "done"