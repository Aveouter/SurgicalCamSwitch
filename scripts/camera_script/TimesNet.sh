export CUDA_VISIBLE_DEVICES=3,4

model_name=TimesNet
# data_20221110.csv data_20220722.csv data_20220729.csv data_20220801.csv data_20230315.csv
# total_data_20221110.csv total_data_20220722.csv total_data_20220729.csv total_data_20220801.csv total_data_20230315.csv
for ty in sequence-out surgery-out ; do
  for datascv in total_data_20221110.csv total_data_20220722.csv total_data_20220729.csv total_data_20220801.csv total_data_20230315.csv; do
    python -u run.py \
      --source_path total_data_20221110.csv/total_data_20220722.csv/total_data_20220729.csv/total_data_20220801.csv/total_data_20230315.csv\
      --testype $ty\
      --f_name result_camera \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/camera \
      --data_path $datascv \
      --model_id camera_60_30\
      --model $model_name \
      --data Camera2\
      --loss CrossEntropyLoss\
      --features MS \
      --freq s\
      --seq_len 60 \
      --lradj rate_on_plateau\
      --label_len 60 \
      --pred_len 30 \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 3901 \
      --dec_in 3901 \
      --c_out 3901 \
      --d_model 256 \
      --d_ff 512 \
      --top_k 5 \
      --des 'Exp' \
      --batch_size 8\
      --itr 1\
      --train_epochs 20 \
      --gpu 0\
      --d_model 128\
      --learning_rate 0.0001\
      --patience 5\
      --dropout 0.2\
      --f_name results_tImenets \
      --camera \
      --target label
  done
done