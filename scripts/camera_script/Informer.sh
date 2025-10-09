export CUDA_VISIBLE_DEVICES=0,2,3,4,5


model_name=Informer

for ty in sequence-out surgery-out; do    #surgery-out #total_data_20221110.csv total_data_20220722.csv 
  for datascv in total_data_20221110.csv total_data_20220722.csv total_data_20220729.csv total_data_20220801.csv total_data_20230315.csv; do
    python -u run.py \
      --source_path total_data_20221110.csv/total_data_20220722.csv/total_data_20220729.csv/total_data_20220801.csv/total_data_20230315.csv\
      --testype $ty\
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/camera \
      --data_path $datascv \
      --model_id camera_120_60 \
      --model $model_name \
      --data Camera2\
      --loss CrossEntropyLoss\
      --features MS \
      --freq s\
      --seq_len 120 \
      --lradj rate_on_plateau\
      --label_len 120 \
      --pred_len 60 \
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
      --train_epochs 10 \
      --gpu 1\
      --d_model 128\
      --learning_rate 0.0001\
      --patience 5\
      --dropout 0.2\
      --f_name results_Informer \
      --camera \
      --target label
  done
done


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/camera \
#   --data_path data_20220801.csv \
#   --model_id camera_5_5\
#   --model $model_name \
#   --data Camera2\
#   --loss CrossEntropyLoss\
#   --features MS \
#   --freq s\
#   --seq_len 20 \
#   --lradj rate_on_plateau\
#   --label_len 20 \
#   --pred_len 5 \
#   --e_layers 1 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 829 \
#   --dec_in 829 \
#   --c_out 829 \
#   --batch_size 8\
#   --des 'Exp' \
#   --itr 1\
#   --train_epochs 100 \
#   --gpu 0\
#   --patience 10\
#   --learning_rate 0.01\
#   --dropout 0.1\
#   --camera \
#   --target label