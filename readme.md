# Requirements
- pytorch
- kagglehub

# Training
python train.py --mode train --epochs 30 --model drop30.pth

python train.py --mode test --model drop30.pth

python train.py --mode predict --model drop30.pth --image "PATH_TO_IMAGE"
