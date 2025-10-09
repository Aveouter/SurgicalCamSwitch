#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 创建保存日志和模型的目录
mkdir -p logs
mkdir -p weight

# 设置通用参数
EPOCHS=5
BATCH_SIZE=64
LR=1e-4
LATENT_DIM=64
EGO_DIR="D:\\LiuXy\\CIGIT\\CameraSwitchingAlgorithm\\surgery_video"
CSV_PATH="data/data_20220722.csv"
CAM_DIRS="data/1 data/2 data/3 data/4 data/5 data/6"

# 模式1：Surgery-out，分别运行每一个手术视频
SURGERIES=("video20220801")

for SURGERY in "${SURGERIES[@]}"; do
    echo -e "\033[1;34m🧪 Running mode=Surgery-out, surgery=$SURGERY\033[0m"
    
    python train_vae.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --latent_dim $LATENT_DIM \
        --mode Surgery-out \
        --surgery $SURGERY \
        --ego_dir "$EGO_DIR" \
        --csv_path $CSV_PATH \
        --cam_dirs $CAM_DIRS \
        --save_path "weight/vae_${SURGERY}.pth" \
        --log_file "logs/train_Surgery-out_${SURGERY}.log"

    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Training failed for surgery=$SURGERY. Check logs/train_Surgery-out_${SURGERY}.log\033[0m"
    else
        echo -e "\033[1;32m✅ Training completed for surgery=$SURGERY\033[0m"
    fi

    sleep 3
done



# # 模式2：Senquence-out（不需要 surgery 参数）
# echo -e "\033[1;34m🧪 Running mode=Senquence-out\033[0m"

# python train_vae.py \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --latent_dim $LATENT_DIM \
#     --mode Senquence-out \
#     --ego_dir "$EGO_DIR" \
#     --csv_path $CSV_PATH \
#     --cam_dirs $CAM_DIRS \
#     --save_path "weight/vae_Sequence-out.pth" \
#     --log_file "logs/train_Sequence-out.log"

# if [ $? -ne 0 ]; then
#     echo -e "\033[1;31m❌ Training failed for mode=Senquence-out\033[0m"
# else
#     echo -e "\033[1;32m✅ Training completed for mode=Senquence-out\033[0m"
# fi
