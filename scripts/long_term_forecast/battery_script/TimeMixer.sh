#export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=32
train_epochs=20
patience=10

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
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --target Temperature\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

