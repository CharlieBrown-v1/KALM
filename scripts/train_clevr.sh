# parameters
time=$(date "+%Y%m%d_%H%M")
pattern_a=0,1,2,3

# paths
CONDA_ENV_PATH=/home/user/anaconda3/envs/kalm
MODEL_PATH=./base_models/llama2-hf-chat-7b
DATA_PATH=./data/clevr_robot.npy
LOG_PATH=./runs/${time}-clevr-${pattern_a}
OUTPUT_PATH=./outputs/${time}-clevr-${pattern_a}

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES

echo "===> Task Starting"

# env setup
source "${CONDA_ENV_PATH}/bin/activate" && \
echo "===> Env Setup"

export NCCL_P2P_DISABLE=1  

# run
echo "===> Run Task at ${time}"
mkdir ./runs
mkdir ./outputs
accelerate launch \
    --config_file configs/ds_clevr.yaml \
    src/train_t2t.py \
    --do_train \
    --do_eval \
    --pretrained_path ${MODEL_PATH} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --evaluation_strategy epoch \
    --eval_delay 1 \
    --save_strategy epoch \
    --ddp_timeout 180 \
    --dataset_path ${DATA_PATH} \
    --split_by index \
    --report_to tensorboard \
    --logging_dir ${LOG_PATH} \
    --logging_strategy steps \
    --logging_steps 20 \
    --output_dir ${OUTPUT_PATH} \
    --pretrain_dir "./pretrain_embed" \
    --pattern_num ${pattern_a} \

echo "===> Task finished"
