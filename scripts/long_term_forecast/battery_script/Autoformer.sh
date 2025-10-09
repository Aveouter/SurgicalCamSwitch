export CUDA_VISIBLE_DEVICES=0,2,3,4,5


model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
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
  --des 'Exp' \
  --itr 1\
  --freq s\
  --use_multi_gpu \
  --setting long_term_forecast_Battery_120_60_Autoformer_Battery_ftMS_sl120_ll120_pl60_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0\
  --target Temperature

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/battery \
#   --data_path final_data-CH41.csv \
#   --model_id Battery_120_60 \
#   --model $model_name \
#   --data Battery \
#   --features MS \
#   --seq_len 120 \
#   --label_len 120 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 50

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/battery \
#   --data_path final_data-CH41.csv \
#   --model_id Battery_120_60 \
#   --model $model_name \
#   --data Battery \
#   --features MS \
#   --seq_len 120 \
#   --label_len 120 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/battery \
#   --data_path final_data-CH41.csv \
#   --model_id Battery_120_60 \
#   --model $model_name \
#   --data Battery \
#   --features MS \
#   --seq_len 120 \
#   --label_len 120 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1