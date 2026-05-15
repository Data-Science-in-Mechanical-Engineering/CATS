#!/usr/local_rwth/bin/zsh
#
#SBATCH --job-name=fl
#SBATCH --output=/work/mf724021/slurm_output/%A_%a.out
#SBATCH --account=p0021919
#SBATCH --nodes=1 # request one node
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexander.graefe@dsme.rwth-aachen.de
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --time=2-23:00:00

# squeue -p c23g -t R,PD --sort=-t,-p -O jobid:10,partition:6,state:10,account:10,username:10,name:65,timeused:12,timelimit:12,priority,tres-alloc:75,reason:34,submittime,starttime

module load GCCcore/.13.3.0
module load Python/3.12.3

source ./venv/bin/activate
python train.py "+run=utsd" "root_dir=/work/mf724021/distributed_transformer" "dataset.dataset_base_path=/hpcwork/p0021919/datasets" "train_mixed_precision=True" "num_gradient_accumulation_steps=4" "dataset.subset_name=UTSD-12G" "optimizer.learning_rate=0.0001" "model.partial_layer_dropout_prob=0.01" "optimizer.weight_decay=0.00"