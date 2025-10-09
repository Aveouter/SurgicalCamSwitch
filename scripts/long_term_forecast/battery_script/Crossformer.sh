export CUDA_VISIBLE_DEVICES=6

model_name=Crossformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/battery \
  --data_path final_data-CH41.csv \
  --model_id Battery_120_60 \
  --model $model_name \
  --data Battery \
  --features MS \
  --seq_len 120 \
  --label_len 120 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --top_k 5 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 4 \
  --itr 1\
  --target Temperature

