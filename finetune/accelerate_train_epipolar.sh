source ~/.venv/bin/activate
which python
export TOKENIZERS_PARALLELISM=false

HDFS_DIR=/mnt/hdfs/zhengguangcong
WORKSPACE=/home/tiger/RealCam-I2V/finetune
cd $WORKSPACE

epipolar_mask_weight_scale=0.5
EXPERIMENT_NAME=RealCam-I2V
loss_weight=10.0
SUB_EXPERIMENT_NAME=CogVideoX1.5-5B-ControlNetXs_EpipolarMaskWeight${epipolar_mask_weight_scale}_headDim128_lossWeight${loss_weight}

ACCELERATE_CONFIG_FILE=${WORKSPACE}/accelerate_config.yaml
PRETRAINED_MODEL_DIR=${WORKSPACE}/pretrained_models
DATA_ROOT=${HDFS_DIR}/data/RealCam-Vid
WANDB_DIR=/home/tiger/wandb/${EXPERIMENT_NAME}/${SUB_EXPERIMENT_NAME}
OUTPUT_DIR=${HDFS_DIR}/checkpoints/${EXPERIMENT_NAME}/${SUB_EXPERIMENT_NAME}
mkdir -p ${WANDB_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PRETRAINED_MODEL_DIR}
export WANDB_DIR=${WANDB_DIR}

ln -sfn ${HDFS_DIR}/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct ${PRETRAINED_MODEL_DIR}/
ln -sfn ${HDFS_DIR}/pretrained_models/zai-org/CogVideoX1.5-5B-I2V ${PRETRAINED_MODEL_DIR}/
ln -sfn ${HDFS_DIR}/pretrained_models/JUGGHM/Metric3D ${PRETRAINED_MODEL_DIR}/
ln -sfn ${HDFS_DIR}/pretrained_models/MuteApo/RealCam-I2V ${PRETRAINED_MODEL_DIR}/

SPLIT=train
# Model Configuration
MODEL_ARGS=(
    --model_path ${PRETRAINED_MODEL_DIR}/CogVideoX1.5-5B-I2V
    --model_name "cogvideox1.5-i2v"
    --model_type "i2v"
    --training_type "controlnetxs"
    --time_sampling_type "truncated_normal"
    --time_sampling_mean 0.8
    --time_sampling_std 0.075
    --keep_aspect_ratio
    --apply_epipolar_mask_weight
    --epipolar_mask_weight_scale ${epipolar_mask_weight_scale}
)

# Output Configuration
OUTPUT_ARGS=(
    --wandb_dir $WANDB_DIR
    --output_dir $OUTPUT_DIR
    --report_to "wandb"
    --tracker_name $EXPERIMENT_NAME
    --sub_tracker_name $SUB_EXPERIMENT_NAME
    --resume_from_checkpoint ${OUTPUT_DIR}/checkpoint-49000
)

# Training Configuration
TRAIN_ARGS=(
    --train_steps 500000
    --batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate 5e-5
    --weight_decay 1e-4
    --mixed_precision "bf16"  # ["no", "fp16"]
    --gradient_checkpointing
    --enable_slicing
    --enable_tiling
    --seed 42
    --loss_weight ${loss_weight}
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 0
    --pin_memory
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 500
    --checkpointing_limit 100
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation
    --validation_dir ${WANDB_DIR}
    --validation_steps 1000
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
    --camera_condition_start_timestep 600
)

# extract video latents of 81x256x448 ; "768//3 x 1360//3 "
DATA_ARGS=(
    --data_root ${DATA_ROOT}
    --cache_root ${DATA_ROOT}/cache
    --metadata_path RealCam-Vid_${SPLIT}.npz
    --enable_align_factor
)


# distribution args for multi-node
master_addr=${master_addr:=$ARNOLD_WORKER_0_HOST}
master_port=${master_port:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
master_addr=${master_addr:=$ARNOLD_EXECUTOR_0_HOST}
master_port=${master_port:=$(echo "$ARNOLD_EXECUTOR_0_PORT" | cut -d "," -f 1)}
node_rank="${node_rank:=$ARNOLD_ID}"

nproc_per_node="${nproc_per_node:=$ARNOLD_WORKER_GPU}"
nnodes="${nnodes:=$ARNOLD_WORKER_NUM}"
nproc_per_node="${nproc_per_node:=$ARNOLD_EXECUTOR_GPU}"
nnodes="${nnodes:=$ARNOLD_EXECUTOR_NUM}"
node_rank="${node_rank:=$ARNOLD_ID}"
trial_id="${trial_id:=$ARNOLD_TRIAL_ID}"
echo ${trial_id} ${node_rank} ${nproc_per_node} ${nnodes} ${master_addr} ${master_port}

# DIST_ARGS=(
#     --config_file $ACCELERATE_CONFIG_FILE
#     --num_machines 1
#     --num_processes 8
#     --machine_rank 0
#     # --main_process_ip localhost
#     # --main_process_port 29500
# )

DIST_ARGS=(
    --config_file $ACCELERATE_CONFIG_FILE
    --num_machines ${nnodes}
    --num_processes $(($nnodes * $nproc_per_node))
    --machine_rank ${node_rank}
    --main_process_ip ${master_addr}
    --main_process_port ${master_port}
)


which accelerate

# accelerate launch "${DIST_ARGS[@]}" train.py \
#     "${MODEL_ARGS[@]}" \
#     "${OUTPUT_ARGS[@]}" \
#     "${DATA_ARGS[@]}" \
#     "${TRAIN_ARGS[@]}" \
#     "${SYSTEM_ARGS[@]}" \
#     "${CHECKPOINT_ARGS[@]}" \
#     "${VALIDATION_ARGS[@]}" \
#     --train_resolution "81x768x1280" \
#     --precompute

# Optional for landscape/portrait joint training
# accelerate launch "${DIST_ARGS[@]}" train.py \
#     "${MODEL_ARGS[@]}" \
#     "${OUTPUT_ARGS[@]}" \
#     "${DATA_ARGS[@]}" \
#     "${TRAIN_ARGS[@]}" \
#     "${SYSTEM_ARGS[@]}" \
#     "${CHECKPOINT_ARGS[@]}" \
#     "${VALIDATION_ARGS[@]}" \
#     --train_resolution "81x1360x768"  \
#     --precompute

accelerate launch ${DIST_ARGS[@]} train.py \
    ${MODEL_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${SYSTEM_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${VALIDATION_ARGS[@]} \
    --train_resolution "81x768x1280"  
    # --allow_switch_hw
