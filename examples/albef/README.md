# ALBEF

[ALBEF](https://arxiv.org/abs/2107.07651) (Li et al., 2021) is a multimodal understanding model that uses a contrastive loss to align unimodal embeddings prior to fusing them via a multimodal transformer encoder. It additionally uses momentum distillation to generate and learn from pseudo-targets using EMA versions of model weights. TorchMultimodal provides example scripts for ALBEF on retrieval (image-text or text-image) and visual question answering tasks.

## Prerequisites

Prior to running any of the MDETR tasks, you should
1) Follow the TorchMultimodal installation instructions in the [README](https://github.com/facebookresearch/multimodal/blob/main/README.md).
2) Install ALBEF requirements via `pip install -r examples/albef/requirements.txt`.

## Retrieval

1) First, download and extract the [COCO](https://cocodataset.org/#download) images and annotations for the task. ALBEF uses the Karpathy splits (see [here](https://arxiv.org/pdf/1412.2306.pdf)) for its retrieval task, so it is sufficient to download just the train and val splits. You will also need to download the custom JSON annotations from the original [ALBEF repo](https://github.com/salesforce/ALBEF).

```
# Download and extract train and val splits from COCO 2014 dataset
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip

# Download and extract annotations
wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz
tar -xvzf data.tar.gz
```

2) Set up the config file to point to the relevant filepaths from (1):

```
# In examples/albef/configs/retrieval.yaml
datamodule_args:
  train_files: ["<my_annotations_root>/coco_train.json"]
  test_files: ["<my_annotations_root>/coco_test.json"]
  image_root: "<my_coco_images_root>"
  ...

# Also make sure the checkpoint root directory exists for saving intermediate checkpoints
training_args:
  ...
  checkpoint_root: <my_checkpoint_root>
```

3) Run the fine-tuning script. From examples/albef directory,

Run on CPU:

```
python finetune_retrieval.py --config configs/retrieval.yaml
```

Run on eight GPUs (single node):
```
torchrun --nproc_per_node=8 finetune_retrieval.py --config ./configs/retrieval.yaml
```


## Visual question answering

1) Download the images and annotations. ALBEF uses a combination of the [VQA V2](https://visualqa.org/download.html) dataset and the [Visual Genome](https://visualgenome.org/api/v0/api_home.html) dataset. As in the retrieval task (see above), the ALBEF repo provides custom annotation files for these two datasets.

First follow step (1) of the retrieval task. VQA V2 just uses COCO images and ALBEF's annotations are all bundled together, so the only remaining object to download is the COCO 2015 test split:

```
wget http://images.cocodataset.org/zips/test2015.zip
unzip test2015.zip
```

After that, all that's left is Visual Genome images.

```
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip
unzip images2.zip

# This will create two separate directories.
# Make sure to put all images in one directory
mv VG_100K_2/* VG_100K/
```

2) Set up the config file to point to the relevant filepaths from (1).

```
# In examples/albef/configs/vqa.yaml
datamodule_args:
  train_files: ["<my_annotations_root>/coco_train.json", "<my_annotations_root>/vg_qa.json", "<my_annotations_root>/vqa_val.json"]
  test_files: ["<my_annotations_root>/vqa_test.json"]
  answer_list: "<my_annotations_root>/answer_list.json"
  vqa_root: "<my_coco_images_root>"
  vg_root: "<my_visual_genome_images_root>"
  ...

# Also make sure the checkpoint root directory exists for saving intermediate checkpoints
training_args:
  ...
  checkpoint_root: <my_checkpoint_root>
```

3) Run the fine-tuning script. From examples/albef directory,

Run on CPU:

```
python finetune_vqa.py --config configs/vqa.yaml
```

Run on eight GPUs (single node):
```
torchrun --nproc_per_node=8 finetune_vqa.py --config ./configs/vqa.yaml
```
