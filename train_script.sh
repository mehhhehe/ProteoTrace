
python train.py \
  --root ./data \
  --model_dir ./models_strong \
  --agg_method mean \
  --hidden_dim 256 \
  --num_layers 3 \
  --batch_size 2048 \
  --num_neighbors 25 15 10 \
  --epochs 30


python train_hybrid.py \
  --root ./data \
  --model_dir ./models_strong \
  --agg_method mean \
  --hidden_dim 256 \
  --num_layers 3 \
  --classifier all


