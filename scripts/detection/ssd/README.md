# Single Shot Multibox Object Detection [1]

[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#object-detection)


## References
1. Wei Liu, et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu, et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.


## Example: SSD-VGG16

### Prepare dataset

- Reference this [page](https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html#sphx-glr-build-examples-datasets-pascal-voc-py) for making VOC dataset, note that only need infernece dataset.

- Reference this [page](https://gluon-cv.mxnet.io/build/examples_datasets/mscoco.html#sphx-glr-build-examples-datasets-mscoco-py) for making COCO2017 dataset, note that only need infernece dataset and data files are already downloaded in /lustre/dataset/COCO2017

### Export models
```
python demo_ssd.py --network=ssd_300_vgg16_atrous_voc
```
Tips:

- you will get a JSON file and a related parameter file
- try other models

### Calibration
```
python calibration.py --model-prefix=ssd_300_vgg16_atrous_voc --data-shape=300 --dataset=voc --calib-mode=naive
```
Tips:

- you will get a quantized JSON file and a related parameter file
- try to set different --num-calib-batch and --calib-mode=entropy to get different int8 accuracy
- try coco models with coco dataset

### Inference

```
# KMP/OMP Settings
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=26

# float32
numactl --physcpubind=0-27 --membind=0 python eval_ssd.py --network=vgg16_atrous --data-shape=300 --batch-size=224 --load-symbol

# int8
numactl --physcpubind=0-27 --membind=0 python eval_ssd.py --network=vgg16_atrous --data-shape=300 --batch-size=224 --load-symbol --quantized
```

Tips:

- try different batch size for performance
- try different combines of OMP_NUM_THREADS and --num-data-workers, the sum of them should be the number of physical cores of a single socket
