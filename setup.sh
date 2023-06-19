conda create -n pr_env python=3.9.4
conda activate pr_env
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
pip install torch torchvision torchaudio
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0