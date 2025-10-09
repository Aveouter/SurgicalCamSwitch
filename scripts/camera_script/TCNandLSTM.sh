#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# === 基本配置 ===
model_name=TimesNet
dates=(20221110 20220722 20220729 20220801 20230315)
backbones=(resnet18)
oj="yolov5s"
models=(TCN Lstm)
types=(sequence-out surgery-out)  # 可改为单一模式

# === 函数：获取backbone维度 ===
get_dim() {
    case "$1" in
        resnet18|resnet34) echo 512 ;;
        resnet50) echo 2048 ;;
        vit_b_16) echo 768 ;;
        *) echo "⚠️ Unknown backbone: $1" >&2; echo 0 ;;
    esac
}

# === 主循环 ===
for model in "${models[@]}"; do
  for ty in "${types[@]}"; do
    for backbone in "${backbones[@]}"; do

      # 拼接输入路径
      input_dir_list=""
      for d in "${dates[@]}"; do
        input_dir_list+=$(printf "merged_%s_%s_video%s.csv/" "$backbone" "$oj" "$d")
      done
      input_dir_list=${input_dir_list%/}  # 去除最后一个斜杠

      # 计算维度
      dim=$(get_dim "$backbone")
      total_dim=$((829 + dim * 6))
      root_path="/baksv/CIGIT/GXN_Liuxy/LenSe/${backbone}_${oj}"

      echo ""
      echo "=== 🔁 Running: Model=$model | Type=$ty | Backbone=$backbone ==="
      echo "Root path     : $root_path"
      echo "Input sources : $input_dir_list"
      echo "Feature dim   : $total_dim"
      echo "---------------------------------------------"

      # 每个 video 执行一次训练
      for d in "${dates[@]}"; do
        datafile="merged_${backbone}_${oj}_video${d}.csv"
        echo "▶️ Running on data file: $datafile"

        python -u run.py \
          --source_path "$input_dir_list" \
          --testype "$ty" \
          --f_name result_camera \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$root_path" \
          --data_path "$datafile" \
          --model_id camera_12_6 \
          --model "$model" \
          --data Camera2 \
          --loss CrossEntropyLoss \
          --features MS \
          --freq s \
          --seq_len 12 \
          --lradj rate_on_plateau \
          --label_len 12 \
          --pred_len 6 \
          --e_layers 1 \
          --d_layers 1 \
          --factor 3 \
          --enc_in $total_dim \
          --dec_in $total_dim \
          --c_out $total_dim \
          --d_model 128 \
          --d_ff 512 \
          --top_k 5 \
          --des 'Exp' \
          --batch_size 8 \
          --itr 1 \
          --train_epochs 20 \
          --gpu 3 \
          --learning_rate 0.0001 \
          --patience 3 \
          --dropout 0.2 \
          --f_name "${model}_${backbone}_${oj}" \
          --camera \
          --target label
        # exit 0  # 仅运行一次，去掉 exit 0 可取消 
      done
    done
  done
done
