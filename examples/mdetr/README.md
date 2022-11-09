# MDETR

[MDETR](https://arxiv.org/abs/2104.12763) (Kamath et al., 2021) is a multimodal reasoning model to detect objects in an image conditioned on a text query. TorchMultimodal provides example scripts for MDETR on phrase grounding and visual question answering tasks.

## Prerequisites

Prior to running any of the MDETR tasks, you should
1) follow the TorchMultimodal installation instructions in the [README](https://github.com/facebookresearch/multimodal/blob/main/README.md).
2) install MDETR requirements via `pip install -r examples/mdetr/requirements.txt`.

## Phrase grounding

In phrase grounding, the objective is to associate noun phrases in the caption of an `(image, text)` pair to regions in the image. Phrase grounding tasks are not straightforward to evaluate in the case where a single phrase refers to multiple distinct boxes in the image, and different papers handle this case differently. One protocol, referred to as the Any-Box Protocol, considers the prediction to be correct based on the maximal IoU value over all ground truth boxes. In this protocol, the pretrained MDETR checkpoint can be evaluated directly on the holdout set without further fine-tuning. We provide a script for evaluation of MDETR on the phrase grounding task using this protocol. For additional details, see Appendix D of the MDETR paper.

### Instructions

First, make sure you have followed the TorchMultimodal installation instructions in the [README](https://github.com/facebookresearch/multimodal/blob/main/README.md).

To run the evaluation script, you will need to download the Flickr30k dataset. This includes images, standard annotations, and additional annotations used by MDETR.

1) Download the Flickr30k images [here](http://shannon.cs.illinois.edu/DenotationGraph/). You will need to fill out the form to request access before receiving the download link in your e-mail.

```
# Download Flickr30k images following the link above, then
tar -xvzf flickr30k-images.tar.gz

# Note that MDETR will expect separate directories for each dataset split, but you can just create symlinks.
ln -s flickr30k-images flickr30k-images/train
ln -s flickr30k-images flickr30k-images/val
ln -s flickr30k-images flickr30k-images/test
```

2) Download the annotations and split mappings from [flickr30k_entities](https://github.com/BryanPlummer/flickr30k_entities).

```
wget https://github.com/BryanPlummer/flickr30k_entities/blob/master/annotations.zip
unzip annotations.zip
wget https://github.com/BryanPlummer/flickr30k_entities/blob/master/train.txt
wget https://github.com/BryanPlummer/flickr30k_entities/blob/master/val.txt
wget https://github.com/BryanPlummer/flickr30k_entities/blob/master/test.txt
```

3) Download and unzip the MDETR custom annotations.

```
wget https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1
tar -xvzf 'mdetr_annotations.tar.gz?download=1'
```

4) Modify the fields in `phrase_grounding.json` based on the locations of the files in (1) - (3). E.g. if root directory for (1)-(3) is /data, you should use

```
{
    "combine_datasets": ["flickr"],
    "combine_datasets_val": ["flickr"],
    "GT_type" : "separate",
    "flickr_img_path" : "/data/flickr30k-images",
   "flickr_dataset_path" : "/data/flickr30k/",
   "flickr_ann_path" : "/data/OpenSource"
  }
```

5) Run the evaluation script:

```
# From REPO_ROOT/examples/mdetr
CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --nproc_per_node=2 phrase_grounding.py --resume https://pytorch.s3.amazonaws.com/models/multimodal/mdetr/pretrained_resnet101_checkpoint.pth --ema --eval --dataset_config phrase_grounding.json
```

## VQA

Coming soon
