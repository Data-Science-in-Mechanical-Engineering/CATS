#!/usr/local_rwth/bin/zsh
#
#SBATCH --job-name=pruning
#SBATCH --output=/work/mf724021/slurm_output/%A_%a.out
#SBATCH --account=p0021919
#SBATCH --nodes=1 # request one node
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexander.graefe@dsme.rwth-aachen.de
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-10:00:00

module load GCCcore/.13.3.0
module load Python/3.12.3

source ./venv/bin/activate
# for df in True False; do
for i in {0..5}; do
    echo "Iteration $i"
    python train.py "+run=monash" "dataset=london_smart_meters_dataset" "root_dir=/work/p0021919/distributed_transformer" "model=verytiny" "dataset.dataset_base_path=/hpcwork/p0021919/datasets" "wandb_project=pruning3" "pruning_step=$i" "do_finetuning=False"

    # python train.py "+run=monash" "dataset=traffic_hourly_dataset" "root_dir=/work/p0021919/distributed_transformer" "model=verytiny" "dataset.dataset_base_path=/hpcwork/p0021919/datasets" "wandb_project=pruning3" "pruning_step=$i" "do_finetuning=False"

    # python train.py "+run=icd" "root_dir=/work/mf724021/distributed_transformer" "model=verytiny" "dataset.dataset_base_path=/hpcwork/p0021919/datasets" "wandb_project=ett_pruning2" "pruning_step=$i" "do_finetuning=False"

    # python train.py "+run=ett" "root_dir=/work/mf724021/distributed_transformer" "model=verytiny" "dataset.dataset_base_path=/hpcwork/p0021919/datasets" "wandb_project=ett_pruning2" "pruning_step=$i" "do_finetuning=False" "dataset.variant=h2"
done
# done