# https://github.com/teticio/audio-diffusion
source /home/lsl/anaconda3/envs/pt/bin/activate pt

audio_type=49 # 0 ~ 49 
input_dir=ESC-50/audio_$audio_type

echo "extract the audio of classification {$audio_type} from dataset ESC-50 ..."

python ESC-50/extract_audio.py \
--audio_type $audio_type

echo "convert audio to images dataset..."

python scripts/audio_to_images.py \
--resolution 64,64 \
--hop_length 1024 \
--input_dir $input_dir \
--output_dir ./output_mel_images

echo "train the model..."

accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
--dataset_name output_mel_images \
--hop_length 1024 \
--output_dir models/ddpm-ema-audio-64-type-$audio_type \
--train_batch_size 16 \
--num_epochs 100 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--lr_warmup_steps 500 \
--mixed_precision no