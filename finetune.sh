export NCCL_TIMEOUT=3600
NCCL_TIMEOUT=3600 NCCL_BLOCKING_WAIT=1 python lerobot/scripts/dps_train.py \
--deepspeed="./ds_zero2.json" \
--policy.type="qwen" \
--policy.num_steps=100 \
--dataset.root="Hephaistos" \
--dataset.repo_id="Hephaistos" \
--wandb.enable=true \
--wandb.project="qwen-pi0-ft-simulated" \
--job_name="0506-qwen-pi0-libero-all_ft-expert-vl-final-layer-df10-wo-img-aug-1st" \
--log_dir="/mnt/wangxiaofa/logs" \
--output_dir="/mnt/wangxiaofa/qwen-pi0-ft-simulated/0506_libero-all-ft-expert-vl-final-layer_df10-wo-img-aug" \
--steps=30_0000 \
--save_freq=10000 \
# --dataset.image_transforms.enable=true