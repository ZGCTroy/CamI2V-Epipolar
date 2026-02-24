HDFS_DIR=/mnt/hdfs/zhengguangcong
HDFS=hdfs://xx/home/xxx/ssd/user/zhengguangcong

WORK_DIR=/home/tiger/RealCam-I2V
cd ${WORK_DIR}


# export http_proxy=xxxxx
# export https_proxy=xxxxx
cd /home/tiger
# curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/.local/bin
sudo cp ${HDFS_DIR}/uv/uv* ~/.local/bin/
sudo chmod +x ~/.local/bin/uv
sudo chmod +x ~/.local/bin/uvx


cd /home/tiger
uv venv --python 3.11 --seed
source ~/.venv/bin/activate

cd ${WORK_DIR}
sudo apt-get -y update
sudo apt-get -y install libgl1-mesa-glx libgl1-mesa-dri xvfb # for ubuntu
export UV_HTTP_TIMEOUT=3600
uv pip install setuptools==65.5.0
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt --no-build-isolation
# uv pip install --force-reinstall -r requirements.txt
uv pip install ${HDFS_DIR}/wheels/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
uv pip install ${HDFS_DIR}/wheels/vsa-0.0.3-cp311-cp311-linux_x86_64.whl
uv pip install ${HDFS_DIR}/wheels/fastvideo_kernel-0.2.5-cp311-cp311-linux_x86_64.whl
uv pip install ${HDFS_DIR}/wheels/magi_attention-1.0.4.post5+g9864120.d20251027-cp311-cp311-linux_x86_64.whl

# unset http_proxy
# unset https_proxy

cd ${WORK_DIR}
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=${TORCH_LIB_PATH}:$LD_LIBRARY_PATH


export NCCL_DEBUG=WARN
pgrep -f wandb | xargs kill -9
pgrep -f train.py | xargs kill -9


bash finetune/${1}.sh



