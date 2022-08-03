#!/bin/sh

#SBATCH --exclusive
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8

RDZV_ENDPOINT=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun python3 -m torch.distributed.run --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_TASK --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDZV_ENDPOINT --max_restarts 0 -m flava.native.train config=flava/native/configs/pretrain_debug.yaml $@
