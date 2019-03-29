python demo.py --export=True --input-pic=./soccer.png --pose-model=simple_pose_resnet50_v1b

python validate_symbolic.py --model-prefix ./simple_pose_resnet50_v1b --num-joints 17 --batch-size 32 --calibration True

python validate_symbolic.py --model-prefix ./simple_pose_resnet50_v1b --num-joints 17 --batch-size 128

python validate_symbolic.py --model-prefix ./simple_pose_resnet50_v1b-naive-quantized --num-joints 17 --batch-size 128

python demo.py --block=True --input-pic=./soccer.png --pose-model=simple_pose_resnet50_v1b-naive-quantized

python demo.py --block=True --input-pic=./soccer.png --pose-model=simple_pose_resnet50_v1b