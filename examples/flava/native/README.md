# Training Instructions

## Prerequisites

- Installed PyTorch 1.13 or nightly 
- `git clone https://github.com/facebookresearch/multimodal`
`cd multimodal/examples`

Configuration presets can be found at: `examples/flava/native/configs`

Using [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html):

**Single node**

`NUM_GPUS=8; torchrun --nproc_per_node=$NUM_GPUS -m flava.native.train config=flava/native/configs/pretrain_debug.yaml`

**Multiple nodes (using slurm)**

Create a `run.slurm` file:

```bash
RDZV_ENDPOINT=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_TASK --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDZV_ENDPOINT --max_restarts 0  -m flava.native.train config=flava/native/configs/pretrain_debug.yaml
$@
```

Run in terminal:

`sbatch --partition=[PARTITION] --nodes=[NUM_NODES] --gpus-per-task=[NUM_GPUS_PER_NODE] run.slurm`
