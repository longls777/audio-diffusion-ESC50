source /home/lsl/anaconda3/envs/pt/bin/activate pt
accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
--dataset_name /home/lsl/projects/audio-diffusion/output_mel_images \
--hop_length 1024 \
--output_dir models/ddpm-ema-audio-64 \
--train_batch_size 16 \
--num_epochs 100 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--lr_warmup_steps 500 \
--mixed_precision no