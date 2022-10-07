# Usage Instructions

This is a lightweight native pytorch implementation to run scaling studies on the FLAVA model. The original code is located at: [`examples/flava/train.py`](https://github.com/facebookresearch/multimodal/blob/main/examples/flava/train.py)

## Prerequisites

- Install torchmultimodal library [from source](https://github.com/facebookresearch/multimodal/blob/main/README.md#building-from-source)
- `cd multimodal/examples`
- `pip install -r flava/requirements.txt`

## Training

### Configuration

Configuration presets for various model sizes can be found at: `examples/flava/native/configs`

Some config settings that are relevant for scaling: (local) `batch_size`, `activation_checkpointing`, `strategy`.

Configs can be overridden through command line, for example: `python -m flava.native.train config=flava/native/configs/pretrain_debug.yaml training.batch_size=8 training.enable_amp=True training.activation_checkpointing=True training.strategy=fsdp`

### Running


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
