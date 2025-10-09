#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export OMP_NUM_THREADS=4  # 限制OpenMP线程数
export MKL_NUM_THREADS=4  # 限制MKL线程数

model_name=PatchTST
dates=(20221110 20220722 20220729 20220801 20230315)
backbones=(resnet18)
oj="yolov5s"

# 函数：根据backbone名返回对应的维度
get_dim() {
    case "$1" in
        resnet18|resnet34) echo 512 ;;
        resnet50) echo 2048 ;;
        vit_b_16) echo 768 ;;
        *) echo "⚠️ 未知backbone: $1" >&2; echo 0 ;;
    esac
}

for ty in sequence-out surgery-out; do
  for backbone in "${backbones[@]}"; do
    # 在 backbone 循环里一次性生成所有日期拼接文件列表
    joined=""
    for d in "${dates[@]}"; do
      joined+=$(printf "merged_%s_%s_video%s.csv/" "$backbone" "$oj" "$d")
    done
    joined=${joined%/}  # 去掉最后的斜杠
    ###
    total_dim=0
    dim=$(get_dim "$backbone")
    total_dim=$((829 + dim*6))
    
    printf "总维度为: %d\n" "$total_dim"
    # 在 backbone 循环中一次性生成 root_path
    root_path=$(printf "/baksv/CIGIT/GXN_Liuxy/LenSe/%s_%s" "$backbone" "$oj")
    echo "Running: date=$d, backbone=$backbone, oj=$oj, type=$ty"
    echo "Root path: $root_path"
    echo "Joined path: $joined"
    for d in "${dates[@]}"; do
      datafile=$(printf "merged_%s_%s_video%s.csv" "$backbone" "$oj" "$d")
      echo "Data file: $datafile"
      python -u run.py \
          --source_path "$joined" \
          --testype "$ty" \
          --f_name result_camera \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$root_path" \
          --data_path "$datafile" \
          --model_id camera\
          --model "$model_name" \
          --data Camera2 \
          --loss CrossEntropyLoss \
          --features MS \
          --freq s \
          --seq_len 120 \
          --lradj rate_on_plateau \
          --label_len 120 \
          --pred_len 60 \
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
          --train_epochs 10 \
          --gpu 0 \
          --learning_rate 0.0001 \
          --patience 3 \
          --dropout 0.2 \
          --f_name "${backbone}_${oj}" \
          --camera \
          --target label
    done
  done
done
