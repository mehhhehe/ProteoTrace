python orchestrate_pipeline.py \
  --root data \
  --model_dir models \
  --agg_method mean \
  --hidden_dim 128 \
  --num_layers 2 \
  --batch_size 256 \
  --num_neighbors 25 10 \
  --epochs 20 \
  --sens_param hidden_dim \
  --sens_values 64 128 256 512



python ./plot_analysis_diagrams.py --root_dir models