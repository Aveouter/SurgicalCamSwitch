# export CUDA_VISIBLE_DEVICES=3,4

# model_name=TimesNet
# # data_20221110.csv/data_20220722.csv/data_20220729.csv/data_20220801.csv/data_20230315.csv
# for ty in sequence-out; do
#   # # 根据ty的值执行不同的操作
#   # if [ "$ty" = "surgery-out" ]; then
#   #   # 如果ty是"surgery-out"，进行特定操作
#   #   source_path = 'image_features_video20220722_with_label.csv/image_features_video20220729_with_label.csv/image_features_video20220801_with_label.csv/image_features_video20221110_with_label.csv/image_features_video20230315_with_label.csv'
#   # elif [ "$ty" = "senqence-out" ]; then
#   #   # 如果ty是"senqence-out"，进行其他操作
#   #   source_path = 'image_features_video20220722_with_label.csv/image_features_video20220729_with_label.csv/image_features_video20220801_with_label.csv/image_features_video20221110_with_label.csv/image_features_video20230315_with_label.csv'
#   # fi
#   for datascv in data_20221110.csv data_20220722.csv data_20220729.csv data_20220801.csv data_20230315.csv; do
#     python -u run.py \
#       --source_path data_20221110.csv/data_20220722.csv/data_20220729.csv/data_20220801.csv/data_20230315.csv\
#       --testype $ty\
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --f_name resultwoV \
#       --root_path ./data/camera \
#       --data_path $datascv \
#       --model_id camera_12_6_image_S_features\
#       --model $model_name \
#       --data Camera2\
#       --loss CrossEntropyLoss\
#       --features MS \
#       --freq s\
#       --seq_len 12 \
#       --lradj rate_on_plateau\
#       --label_len 12 \
#       --pred_len 6 \
#       --e_layers 1 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 3073 \
#       --dec_in 3073 \
#       --c_out 3073 \
#       --d_model 256 \
#       --d_ff 512 \
#       --top_k 5 \
#       --des 'Exp' \
#       --batch_size 64\
#       --itr 1\
#       --train_epochs 10 \
#       --gpu 1\
#       --d_model 128\
#       --learning_rate 0.0001\
#       --patience 5\
#       --dropout 0.3\
#       --camera \
#       --target label
#   done
# done

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
      --model_id camera_12_6\
      --model $model_name \
      --data Camera2\
      --loss CrossEntropyLoss\
      --features MS \
      --freq s\
      --seq_len 12 \
      --lradj rate_on_plateau\
      --label_len 12 \
      --pred_len 6 \
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
      --camera \
      --target label
  done
done