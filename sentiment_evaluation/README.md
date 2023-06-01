# Sentiment evaluation on HPC

The script ```sentiment_eval_pilot.py``` is made for the DTU HPC, but should work on local machines as well with a few customizations.

## Running the script
The script takes 2 terminal arguments such that a range can be set to index the dataframe. \
Example:
```
python sentiment_eval_pilot.py 0 40000
```
This can either be run in one of the interactive GPU nodes on sent as a batch job.
### Batch job:
``` submit_pilot_subset_1.sh``` is an example of one of the shell scripts used to queue one of these tasks.
## Requirements for virtual enviroment on the HPC
Below are the packages used to run the virtual enviroment for this script.
```
certifi==2023.5.7
cffi==1.15.1
charset-normalizer==3.1.0
cmake==3.26.3
cryptography==41.0.0
filelock==3.12.0
fsspec==2023.5.0
huggingface-hub==0.14.1
idna==3.4
Jinja2==3.1.2
lit==16.0.5
MarkupSafe==2.1.2
mpmath==1.3.0
mypy-extensions==1.0.0
networkx==3.1
numpy==1.24.3
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.2.10.91
nvidia-cusolver-cu11==11.4.0.1
nvidia-cusparse-cu11==11.7.4.91
nvidia-nccl-cu11==2.14.3
nvidia-nvtx-cu11==11.7.91
packaging==23.1
pandas==2.0.2
pycparser==2.21
pyOpenSSL==23.2.0
pyre-extensions==0.0.29
python-dateutil==2.8.2
pytz==2023.3
PyYAML==6.0
regex==2023.5.5
requests==2.31.0
scipy==1.10.1
six==1.16.0
sympy==1.12
tokenizers==0.13.3
torch==2.0.1
tqdm==4.65.0
transformers==4.29.2
triton==2.0.0
typing-inspect==0.9.0
typing_extensions==4.6.2
tzdata==2023.3
urllib3==1.26.6
xformers==0.0.20
```