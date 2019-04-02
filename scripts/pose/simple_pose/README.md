# Example: simple_pose_resnet50_v1b

## Prepare dataset

- Reference this [page](https://gluon-cv.mxnet.io/build/examples_datasets/mscoco.html#sphx-glr-build-examples-datasets-mscoco-py) for making COCO2017 dataset, note that only need inference dataset and data files are already downloaded in /lustre/dataset/COCO2017

## Export Models
```
python demo.py --export=True  --pose-model=simple_pose_resnet50_v1b
```
## Calibration
```
python validate_symbolic.py --model-prefix ./simple_pose_resnet50_v1b --num-joints 17 --batch-size 32 --calibration True
```
## FP32 Inference
```
# KMP/OMP Settings
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=?

export MXNET_SUBGRAPH_BACKEND=MKLDNN

numactl --physcpubind=0-27 --membind=0 python validate_symbolic.py --model-prefix ./simple_pose_resnet50_v1b --num-joints 17 --batch-size 128 --num-data-workers=?
```
## INT8 Inference
```
# KMP/OMP Settings
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=?

numactl --physcpubind=0-27 --membind=0 python validate_symbolic.py --model-prefix ./simple_pose_resnet50_v1b-naive-quantized --num-joints 17 --batch-size 128 --num-data-workers=?
```
## Optional: Demo
```
python demo.py --block=True --input-pic=./soccer.png --pose-model=simple_pose_resnet50_v1b-naive-quantized

python demo.py --block=True --input-pic=./soccer.png --pose-model=simple_pose_resnet50_v1b
```
