# Setup
- experimente wurden in ubuntu 24.04. ausgeführt
- clone this repository: git clone https://github.com/lremane/HoiTransformer.git
- install miniconda: https://docs.anaconda.com/miniconda/install/#quick-command-line-install
- activate miniconda: source ~/miniconda3/bin/activate
- create HoiTransformer environment vom environment.yml file aus dem Root-Verzeichnis der Projektes: conda env create --name HoiTransformer --file environment.yml
- acitivate HoiTransformer environment: conda activate HoiTransformer
- place the training and test data in the direcoty: data

## training
- place in root directory pretrained model resnet-model: link to modells
- with: pip install gdown && gdown https://drive.google.com/uc?id=1-WQnnTHB7f7X2NpqPVqIO6tvWN6k1Ot8
- download detr modells: cd data/detr_coco && wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
- run finetune.py: python finetune.py --epochs=250 --lr_drop=10 --dataset_file=hico --batch_size=2 --backbone=resnet50 --lr=0.00001 --lr_backbone=0.00001 --resume=res50_hico.pth

## testing
python test.py --backbone=resnet50 --batch_size=1 --dataset_file=hico --log_dir=./ --model_path=checkpoint/p_202501261602//checkpoint.pth --epoch=164



## visualize results
- either visualize a specific image with: --image_name or all images in a directoy with --image_directory. the first will be preferred if both are provided
- run: python test_on_images.py --image_directory=<directory_name_with_images> --backbone=resnet50 --batch_size=1 --log_dir=<output_directory> --model_path=<model_name.pth> --dataset=hico

Customize visualization thresholds with the following optional arguments:
- **`--hoi_th`** (default: `0.6`): Threshold for visualizing Human-Object Interactions.  
- **`--human_th`** (default: `0.4`): Threshold for visualizing humans.  
- **`--object_th`** (default: `0.4`): Threshold for visualizing objects.  
- **`--top_k`** (default: `5`): Maximum number of interactions to visualize.  

Adjust these values as needed when running the script. Example:  
```bash
python test_on_images.py --hoi_th=0.5 --image_path=<directory_name_with_images> --backbone=resnet50 --batch_size=1 --log_dir=test_results --model_path=<model_name.pth> --dataset=hico
```


# HOI Transformer
Code for CVPR 2021 accepted paper [End-to-End Human Object Interaction Detection with HOI Transformer](https://arxiv.org/abs/2103.04503).

This method also won 2nd Place Award in HOI Challenge in [Person In Context](http://www.picdataset.com/challenge/leaderboard/pic2021) in CVPR Workshop 2021.

<div align="center">
  <img src="data/architecture.png" width="900px" />
</div>


## TODO list

- [x] HICO-DET Code & DataSet
- [x] V-COCO Code & DataSet
- [x] HOI-A Code & DataSet
- [x] HICO-DET [Res50](https://drive.google.com/file/d/1-WQnnTHB7f7X2NpqPVqIO6tvWN6k1Ot8/view?usp=sharing) Res101
- [x] V-COCO Res50 [Res101](https://drive.google.com/file/d/1HZH3fUpiou2-f91_OvHnTX6feZNqxHa7/view?usp=sharing)
- [x] HOI-A [Res50](https://drive.google.com/file/d/1bNrFQ6a8aKBzwWc0MAdG2f24StMP9lhY/view?usp=sharing) Res101
- [x] Swin-B Code


## Performance
|Model|HICO-DET (Full/Rare/NonRare)|V-COCO|HOI-A|
|---|---|---|---|
|Res50|28.92/19.83/31.64|51.15|73.23|


## Reproduction

We recomend you to setup in the following steps:

1.Clone the repo.
```
git clone https://github.com/bbepoch/HoiTransformer.git
```

2.Download the MS-COCO pretrained [DETR](https://github.com/facebookresearch/detr) model.
```bash
cd data/detr_coco && bash download_model.sh
```

3.Download the annotation files for HICO-DET, V-COCO and HOI-A.
```bash
cd data && bash download_annotations.sh
```

4.Download the image files for [HICO-DET](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk), [V-COCO](https://cocodataset.org/#download) and [HOI-A](https://drive.google.com/drive/folders/15xrIt-biSmE9hEJ2W6lWlUmdDmhatjKt). Instead, we provide a [script](data/download_images.sh) to get all of them. A required directory structure is:

        HoiTransformer/
        ├── data/
        │   ├── detr_coco/
        │   ├── hico/
        │   │   ├── eval/
        │   │   └── images/
        │   │       ├── train2015/
        │   │       └── test2015/
        │   ├── hoia/
        │   │   ├── eval/
        │   │   └── images/
        │   │       ├── trainval/
        │   │       └── test/
        │   └── vcoco/
        │       ├── eval/
        │       └── images/
        │           ├── train2014/
        │           └── val2014/
        ├── datasets/
        ├── models/
        ├── tools/
        ├── util/
        ├── engin.py
        ├── main.py
        └── test.py

5.OPTIONAL SETTINGS. When the above subdirectories in 'data' are all ready, you can train a model on any one of the three benchmarks. But before that, we highly recommend you to move the whole folder 'data' to another place on your computer, e.g. '/home/hoi/data', and only put a soft link named 'data' under 'HoiTransformer'.
```bash
# Optional but recommended to separate data from code.
mv data /home/hoi/
ln -s /home/hoi/data data
```

6.Train a model.
```
# Train on HICO-DET.
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --epochs=150 --lr_drop=110 --dataset_file=hico --batch_size=2 --backbone=resnet50

# Train on HOI-A.
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --epochs=150 --lr_drop=110 --dataset_file=hoia --batch_size=2 --backbone=resnet50

# Train on V-COCO.
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --epochs=150 --lr_drop=110 --dataset_file=vcoco --batch_size=2 --backbone=resnet50

# Training longer can get even better performance.
```

7.Test a model.
```
python3 test.py --backbone=resnet50 --batch_size=1 --dataset_file=hico --log_dir=./ --model_path=your_model_path
```


## Annotations

We propose a new annotation format 'ODGT' which is much easier to understand, and we have provided annotation files for all the existing benchmarks, i.e. HICO-DET, HOI-A, V-COCO, so you don't have to know how to get it, just use it. The core structure of 'ODGT' format is:
```
{
    file_name: XXX.jpg,
    width: image width,
    height: image height,
    gtboxes: [
        {
            box: [x, y, w, h],
            tag: object category name,
        },
        ...
    ],
    hoi: [
        {
            subject_id: human box index in gtboxes,
            object_id: object box index in gtboxes,
            interaction: hoi category name,
        },
        ...
    ],
}
```


## Citation

```
@inproceedings{zou2021_hoitrans,
  author = {Zou, Cheng and Wang, Bohan and Hu, Yue and Liu, Junqi and Wu, Qian and Zhao, Yu and Li, Boxun and Zhang, Chenguang and Zhang, Chi and Wei, Yichen and Sun, Jian},
  title = {End-to-End Human Object Interaction Detection with HOI Transformer},
  booktitle={CVPR},
  year = {2021},
}
```


## Acknowledgement
We sincerely thank all previous works, especially [DETR](https://github.com/facebookresearch/detr), [PPDM](https://github.com/YueLiao/PPDM), [iCAN](https://github.com/vt-vl-lab/iCAN), for some of the codes are built upon them.

