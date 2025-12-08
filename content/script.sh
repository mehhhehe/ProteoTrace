python sensitivity_analysis.py \
  --root ./data \
  --model_dir ./models \
  --agg_method mean \
  --param hidden_dim --values 32 64 128 256 512 \
  --hidden_dim 128 \
  --num_layers 3 \
  --num_neighbors 25 15 10 \
  --batch_size 1024 \
  --epochs 20


python ./plot_analysis_diagrams.py --root_dir models