#!/usr/local_rwth/bin/zsh
#
#SBATCH --job-name=tsfm_train
#SBATCH --output=/work/mf724021/slurm_output/%A_%a.out
#SBATCH --account=p0021919
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexander.graefe@dsme.rwth-aachen.de
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=2-00:00:00

# squeue -p c23g -t R,PD --sort=-t,-p -O jobid:10,partition:6,state:10,account:10,username:10,name:65,timeused:12,timelimit:12,priority,tres-alloc:75,reason:34,submittime,starttime

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=12355
echo "MASTER_PORT="$MASTER_PORT

module load GCCcore/.13.3.0
module load Python/3.12.3

source ./venv/bin/activate
echo "CUDA_VISIBLE_DEVICES="=$CUDA_VISIBLE_DEVICES
srun python train.py "+run=utsd" "root_dir=/work/mf724021/distributed_transformer" "dataset.dataset_base_path=/hpcwork/p0021919/datasets"  "model=base" "do_distributed_training=True" "train_mixed_precision=True" "num_gradient_accumulation_steps=1" "optimizer.learning_rate=0.0001" "model.partial_layer_dropout_prob=0.0" "optimizer.weight_decay=0.00"
