# Image Classification on ImageNet

Please refer to [GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#image-classification)
for available pretrained models, training hyper-parameters, etc.

## Example: ResNet50_v1

### Export models
```
python demo_classification.py --network=resnet50_v1
```
Tips:

- you will get a JSON file and a related parameter file
- try resnet101 and inception-v3
- note that the input size of inception-v3 is 3x299x299

### Calibration
```
python calibration.py --model-prefix=resnet50_v1 --data-dir=/lustre/dataset/mxnet/imagenet --data-shape=224 --calib-mode=naive
```
Tips:

- you will get a quantized JSON file and a related parameter file
- try to set different --num-calib-batch and --calib-mode=entropy to get different int8 accuracy

### Inference

```
# KMP/OMP Settings
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=18

# float32
numactl --physcpubind=0-27 --membind=0 python verify_pretrained.py --model=resnet50_v1 --num-inference-batches=100 --batch-size=128 --load-symbol --num-data-workers=10 --rec-dir=/lustre/dataset/mxnet/imagenet/

# int8
numactl --physcpubind=0-27 --membind=0 python verify_pretrained.py --model=resnet50_v1-quantized --num-inference-batches=100 --batch-size=128 --load-symbol --num-data-workers=10 --rec-dir=/lustre/dataset/mxnet/imagenet/
```

Tips:

- run 50000 images for accuracy(you can set --num-inference-batches=500 --batch-size=100)
- try different batch size for performance
- try different combines of OMP_NUM_THREADS and --num-data-workers, the sum of them should be the number of physical cores of a single socket