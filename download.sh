# download
# export http_proxy=xxxx
# export https_proxy=xxxx
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download zai-org/CogVideoX1.5-5B-I2V \
  --local-dir /mnt/hdfs/zhengguangcong/pretrained_models/zai-org/CogVideoX1.5-5B-I2V

huggingface-cli download MuteApo/RealCam-I2V \
  --local-dir /mnt/hdfs/zhengguangcong/pretrained_models/MuteApo/RealCam-I2V

huggingface-cli download JUGGHM/Metric3D \
  --local-dir /mnt/hdfs/zhengguangcong/pretrained_models/JUGGHM/Metric3D


ln -sfn /mnt/hdfs/zhengguangcong/pretrained_models/MuteApo/RealCam-I2V /home/tiger/RealCam-I2V/checkpoints/RealCam-I2V
ln -sfn /mnt/hdfs/zhengguangcong/pretrained_models/zai-org/CogVideoX1.5-5B-I2V /home/tiger/RealCam-I2V/checkpoints/CogVideoX1.5-5B-I2V



