task="cifar10cnn_50epoch"
study_name=$task
storage="sqlite:///$task.sqlite3"
direction="minimize"

gpu=4
export CUDA_VISIBLE_DEVICES=$gpu
nohup python test.py $study_name $storage $direction >> nohup_${task}_$gpu.out &
sleep 10

gpu=5
export CUDA_VISIBLE_DEVICES=$gpu
nohup python test.py $study_name $storage $direction >> nohup_${task}_$gpu.out &

gpu=6
export CUDA_VISIBLE_DEVICES=$gpu
nohup python test.py $study_name $storage $direction >> nohup_${task}_$gpu.out &

gpu=7
export CUDA_VISIBLE_DEVICES=$gpu
nohup python test.py $study_name $storage $direction >> nohup_${task}_$gpu.out &
