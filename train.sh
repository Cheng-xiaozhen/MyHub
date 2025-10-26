CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  training_scripts/train.py \
  --config config/resnet_train.yaml
