err_report() {
    echo "Error on line $1"
}

trap 'err_report $LINENO' ERR

# A comma-separated list of parameter server processes.
PS_HOSTS="localhost:4000"


# A comma-separated list of worker processes.
WORKER_HOSTS="localhost:5000,localhost:5001,localhost:5002,localhost:5004"

# Where the Swivel training data is located.  All processes must be able to read
# from this directory, so it ought to be a network filesystem if you're running
# on multiple servers.
INPUT_BASE_PATH="/home/ubuntu/swivel/swivel/preprocessed"

# Where the output and working directory is located.
OUTPUT_BASE_PATH="/home/ubuntu/swivel/swivel/embedding"

# Location of evaluation data, if you want to observe evaluation while training.
EVAL_BASE_PATH="/home/ubuntu/swivel/swivel/eval"

ARGS="--ps_hosts ${PS_HOSTS}
--worker_hosts ${WORKER_HOSTS}
--input_base_path ${INPUT_BASE_PATH}
--output_base_path ${OUTPUT_BASE_PATH}
--eval_base_path ${EVAL_BASE_PATH} --num_epochs 5"

echo "started"
# This configuration is for a two-GPU machine.  It starts four worker
# processes, two for each GPU.
python swivel.py --job_name ps --task_index 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/ps.0 &
python swivel.py --job_name worker --task_index 0 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.0 &
python swivel.py --job_name worker --task_index 1 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.1 &
python swivel.py --job_name worker --task_index 2 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.2 &
python swivel.py --job_name worker --task_index 3 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.3 &
python swivel.py --job_name worker --task_index 4 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.4 &
python swivel.py --job_name worker --task_index 5 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.5 &
python swivel.py --job_name worker --task_index 6 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.6 &
python swivel.py --job_name worker --task_index 7 --gpu_device 0 ${ARGS} >& /home/ubuntu/swivel/swivel/egpu_output/worker.7 &

echo "started 8 workers"
# Perhaps there is a more clever way to clean up the parameter server once all
# the workers are done.
wait %2 %3 %4 %5
kill %1
