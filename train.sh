CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=3 \
  -m MyHub.training_scripts.train \
  --config MyHub/config/convnext_train.yaml \
