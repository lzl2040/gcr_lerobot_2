export NCCL_TIMEOUT=3600
NCCL_TIMEOUT=3600 NCCL_BLOCKING_WAIT=1 python lerobot/scripts/dps_train.py \
--deepspeed="./ds_zero2.json" \
--policy.type="qwen" \
--policy.num_steps=100 \
--dataset.root="Hephaistos" \
--dataset.repo_id="Hephaistos" \
--wandb.enable=true \
--wandb.project="qwen-pi0-ft-simulated" \
--job_name="0504-qwen-pi0-libero-all_exp-only-random-order-false-1st" \
--log_dir="/mnt/wangxiaofa/logs" \
--output_dir="/mnt/wangxiaofa/qwen-pi0-ft-simulated/0504_libero-all-exponly_df100-random-order-false" \
--steps=30_0000 \
--save_freq=50000 \
--dataset.image_transforms.enable=true