### 系统
```shell
ftan) ubuntu@ubuntu:~/FTAN$ uname -a 
Linux ubuntu 6.8.0-57-generic #59~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Mar 19 17:07:41 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
(ftan) ubuntu@ubuntu:~/FTAN$ cat /etc/os-release 
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```
### cuda 版本
```shell
(ftan) ubuntu@ubuntu:~/FTAN$ nvidia-smi 
Sun Apr 27 15:38:23 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.58.02              Driver Version: 555.58.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A6000               Off |   00000000:01:00.0  On |                  Off |
| 52%   77C    P0            187W /  300W |    5362MiB /  49140MiB |     85%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2432      G   /usr/lib/xorg/Xorg                            224MiB |
|    0   N/A  N/A    158858      C   ...untu/anaconda3/envs/ftan/bin/python       5118MiB |
+-----------------------------------------------------------------------------------------+
(ftan) ubuntu@ubuntu:~/FTAN$ nvcc --version 
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```
### 安装python环境
```shell
conda create -n ftan python=3.8
conda activate ftan
```
### 环境依赖
```shell
(ftan) ubuntu@ubuntu:~/FTAN/mmsegmentation$ conda list 
# packages in environment at /home/ubuntu/anaconda3/envs/ftan:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
addict                    2.4.0                    pypi_0    pypi
aliyun-python-sdk-core    2.16.0                   pypi_0    pypi
aliyun-python-sdk-kms     2.16.5                   pypi_0    pypi
bzip2                     1.0.8                h4bc722e_7    conda-forge
ca-certificates           2025.1.31            hbd8a1cb_1    conda-forge
certifi                   2025.4.26                pypi_0    pypi
cffi                      1.17.1                   pypi_0    pypi
charset-normalizer        3.4.1                    pypi_0    pypi
click                     8.1.8                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
contourpy                 1.1.1                    pypi_0    pypi
crcmod                    1.7                      pypi_0    pypi
cryptography              44.0.2                   pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
einops                    0.8.1                    pypi_0    pypi
filelock                  3.14.0                   pypi_0    pypi
fonttools                 4.57.0                   pypi_0    pypi
fsspec                    2025.3.0                 pypi_0    pypi
ftfy                      6.2.3                    pypi_0    pypi
huggingface-hub           0.30.2                   pypi_0    pypi
idna                      3.10                     pypi_0    pypi
importlib-metadata        8.5.0                    pypi_0    pypi
importlib-resources       6.4.5                    pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
jmespath                  0.10.0                   pypi_0    pypi
kiwisolver                1.4.7                    pypi_0    pypi
ld_impl_linux-64          2.43                 h712a8e2_4    conda-forge
libffi                    3.4.6                h2dba641_1    conda-forge
libgcc                    14.2.0               h767d61c_2    conda-forge
libgcc-ng                 14.2.0               h69a702a_2    conda-forge
libgomp                   14.2.0               h767d61c_2    conda-forge
liblzma                   5.8.1                hb9d3cd8_0    conda-forge
liblzma-devel             5.8.1                hb9d3cd8_0    conda-forge
libnsl                    2.0.1                hd590300_0    conda-forge
libsqlite                 3.49.1               hee588c1_2    conda-forge
libuuid                   2.38.1               h0b41bf4_0    conda-forge
libxcrypt                 4.4.36               hd590300_1    conda-forge
libzlib                   1.3.1                hb9d3cd8_2    conda-forge
markdown                  3.7                      pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.7.5                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mmcls                     0.25.0                   pypi_0    pypi
mmcv-full                 1.6.2                    pypi_0    pypi
mmengine                  0.10.7                   pypi_0    pypi
mmsegmentation            0.29.0                   pypi_0    pypi
model-index               0.1.11                   pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
ncurses                   6.5                  h2d0b736_3    conda-forge
networkx                  3.1                      pypi_0    pypi
ninja                     1.11.1.4                 pypi_0    pypi
numpy                     1.24.4                   pypi_0    pypi
nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.8.93                  pypi_0    pypi
nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
opencv-python             4.11.0.86                pypi_0    pypi
opendatalab               0.0.10                   pypi_0    pypi
openmim                   0.3.9                    pypi_0    pypi
openssl                   3.5.0                h7b32b05_0    conda-forge
openxlab                  0.1.2                    pypi_0    pypi
ordered-set               4.1.0                    pypi_0    pypi
oss2                      2.17.0                   pypi_0    pypi
packaging                 24.2                     pypi_0    pypi
pandas                    2.0.3                    pypi_0    pypi
pillow                    10.4.0                   pypi_0    pypi
pip                       22.1.2             pyhd8ed1ab_0    conda-forge
platformdirs              4.3.6                    pypi_0    pypi
prettytable               3.11.0                   pypi_0    pypi
psutil                    7.0.0                    pypi_0    pypi
pycparser                 2.22                     pypi_0    pypi
pycryptodome              3.22.0                   pypi_0    pypi
pygments                  2.19.1                   pypi_0    pypi
pyparsing                 3.1.4                    pypi_0    pypi
python                    3.8.20          h4a871b0_2_cpython    conda-forge
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2023.4                   pypi_0    pypi
pyyaml                    6.0.2                    pypi_0    pypi
readline                  8.2                  h8c095d6_2    conda-forge
regex                     2024.11.6                pypi_0    pypi
requests                  2.28.2                   pypi_0    pypi
rich                      13.4.2                   pypi_0    pypi
safetensors               0.5.3                    pypi_0    pypi
scipy                     1.10.1                   pypi_0    pypi
setuptools                75.3.2                   pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
sympy                     1.13.3                   pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
termcolor                 2.4.0                    pypi_0    pypi
thop                      0.1.1-2209072238          pypi_0    pypi
timm                      1.0.15                   pypi_0    pypi
tk                        8.6.13          noxft_h4845f30_101    conda-forge
tomli                     2.2.1                    pypi_0    pypi
torch                     1.11.0+cu115             pypi_0    pypi
torchaudio                0.11.0+cu115             pypi_0    pypi
torchvision               0.12.0+cu115             pypi_0    pypi
tqdm                      4.65.2                   pypi_0    pypi
triton                    3.0.0                    pypi_0    pypi
typing-extensions         4.13.2                   pypi_0    pypi
tzdata                    2025.2                   pypi_0    pypi
urllib3                   1.26.20                  pypi_0    pypi
wcwidth                   0.2.13                   pypi_0    pypi
wheel                     0.45.1             pyhd8ed1ab_0    conda-forge
xz                        5.8.1                hbcc6ac9_0    conda-forge
xz-gpl-tools              5.8.1                hbcc6ac9_0    conda-forge
xz-tools                  5.8.1                hb9d3cd8_0    conda-forge
yapf                      0.43.0                   pypi_0    pypi
zipp                      3.20.2                   pypi_0    pypi
```
### 安装依赖
```shell
pip install -r requirements.txt
```

### 下载数据库集
根据文档[LEVIR-CD](https://opendatalab.com/OpenDataLab/LEVIR-CD/cli/main),[DSIFN-CD](https://opendatalab.com/OpenDataLab/DSIFN-CD/tree/main)分别下载数据集
下载时间比较长
```
pip install openxlab 

Access Key: x7njkqbolvw8wqm2gwmr
Secret Key: vw14jwxn6okaqrnedyknjnjpbl582xebqv3dzlyb
openxlab dataset get --dataset-repo OpenDataLab/DSIFN-CD
openxlab dataset get --dataset-repo OpenDataLab/LEVIR-CD

```
### 下载代码
```shell
git clone https://github.com/kafkaqin/FTAN
```
### 数据预处理
在目录 data_preparation 执行 `levir_cd_256.py脚本，然后再执行 generate_list_txt.py` 注意需要修改数据目录

### 训练
```shell
python main-cd.py
```
### 推理
```shell
python demo_LEVIR.py
```

### 排查问题 
1. mvcc安装问题 
执行下面的命令 查看自己环境安装pytorch版本和cuda版本
```shell
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```
需要通过 通过执行下面命令安装mmcv,相关文档[mmcv](https://mmcv.readthedocs.io/zh-cn/2.x/get_started/installation.html)
```shell
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html
pip install mmsegmentation

```
2. `RuntimeError: Unable to find a valid cuDNN algorithm to run convolutio`
导致这个问题 是因为batch size设置太大导致的,相关链接[RuntimeError: Unable to find a valid cuDNN algorithm to run convolutio](https://zhuanlan.zhihu.com/p/515897782)
3. 找不到mmcv的print_log,报错如下文,是因为mmcv版本对不上
```shell
  File "/home/ubuntu/anaconda3/envs/ftan/lib/python3.8/site-packages/mmcv/parallel/__init__.py", line 5, in <module>
    from .distributed import MMDistributedDataParallel
  File "/home/ubuntu/anaconda3/envs/ftan/lib/python3.8/site-packages/mmcv/parallel/distributed.py", line 8, in <module>
    from mmcv import print_log
ImportError: cannot import name 'print_log' from 'mmcv'
```
4. 报`No module named 'mmseg.ops'` 错误是因为安装的版本太高导致，要安装mmsegmentation==0.29.0版本
```shell
/home/ubuntu/anaconda3/envs/ftan/lib/python3.8/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
Traceback (most recent call last):
  File "main_cd.py", line 3, in <module>
    from models.trainer import *
  File "/home/ubuntu/FTAN/models/trainer.py", line 6, in <module>
    from models.networks import *
  File "/home/ubuntu/FTAN/models/networks.py", line 13, in <module>
    from models.SegNext import SegNext_diffV1
  File "/home/ubuntu/FTAN/models/SegNext.py", line 11, in <module>
    from models.Decoder import ChangeNeXtDecoder
  File "/home/ubuntu/FTAN/models/Decoder.py", line 7, in <module>
    from mmseg.ops import resize
ModuleNotFoundError: No module named 'mmseg.ops'
```
### requirements.txt文件内容
```shell
addict==2.4.0
aliyun-python-sdk-core==2.16.0
aliyun-python-sdk-kms==2.16.5
certifi==2025.4.26
cffi==1.17.1
charset-normalizer==3.4.1
click==8.1.8
colorama==0.4.6
contourpy==1.1.1
crcmod==1.7
cryptography==44.0.2
cycler==0.12.1
einops==0.8.1
filelock==3.14.0
fonttools==4.57.0
fsspec==2025.3.0
ftfy==6.2.3
huggingface-hub==0.30.2
idna==3.10
importlib_metadata==8.5.0
importlib_resources==6.4.5
Jinja2==3.1.6
jmespath==0.10.0
kiwisolver==1.4.7
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.7.5
mdurl==0.1.2
mmcls==0.25.0
mmcv-full==1.6.2
mmengine==0.10.7
# Editable install with no version control (mmsegmentation==0.29.0)
-e /home/ubuntu/anaconda3/envs/ftan/lib/python3.8/site-packages
model-index==0.1.11
mpmath==1.3.0
networkx==3.1
ninja==1.11.1.4
numpy==1.24.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.1.105
opencv-python==4.11.0.86
opendatalab==0.0.10
openmim==0.3.9
openxlab==0.1.2
ordered-set==4.1.0
oss2==2.17.0
packaging==24.2
pandas==2.0.3
pillow==10.4.0
platformdirs==4.3.6
prettytable==3.11.0
psutil==7.0.0
pycparser==2.22
pycryptodome==3.22.0
Pygments==2.19.1
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2023.4
PyYAML==6.0.2
regex==2024.11.6
requests==2.28.2
rich==13.4.2
safetensors==0.5.3
scipy==1.10.1
six==1.17.0
sympy==1.13.3
tabulate==0.9.0
termcolor==2.4.0
thop==0.1.1.post2209072238
timm==1.0.15
tomli==2.2.1
torch==1.11.0+cu115
torchaudio==0.11.0+cu115
torchvision==0.12.0+cu115
tqdm==4.65.2
triton==3.0.0
typing_extensions==4.13.2
tzdata==2025.2
urllib3==1.26.20
wcwidth==0.2.13
yapf==0.43.0
zipp==3.20.2
```