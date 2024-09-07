# YOLO-Light

YOLO-Light: Automatic Lightweight YOLO Generation for Object Detection Tasks in Different Scenarios through NeuroEvolution

## Requirement

```
# conda environment
$ conda create -n yl python=3.12
$ conda activate yl
$ conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics

# modify ultralytics for YOLOv10 NeuroEvolution
$ cd <path to> naconda3/envs/yl/lib/python3.12/site-packages/ultralytics/nn/modules/
$ nano block.py
## modify following line in PSA module:
## self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64 if self.c // 64 > 0 else 1)
```

## Usage

```
# create dataset root directory
$ mkdir data
## download Roboflow 100 dataset into data folder

# modify evolution state file

# evolve yolo
$ python evolve_yolo.py
```

