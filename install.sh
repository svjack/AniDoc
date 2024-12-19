

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install diffusers==0.24.0 transformers==4.27.0 xformers==0.0.16 imageio==2.27.0 decord==0.6.0
pip install huggingface_hub==0.24.7

pip install einops
pip install triton==2.1.0
pip install opencv-python
pip install av scipy
pip install accelerate==0.27.2

pip install colorlog
pip install pyparsing==3.0.9
pip install gradio==3.50.2
pip install omegaconf
pip install scikit-image


cd cotracker && python setup.py install && cd ../ 
pip install kornia
pip install moviepy