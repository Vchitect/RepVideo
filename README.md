conda create -n RepVid python==3.10
conda activate RepVid
pip install -r requirements.txt


mkdir ckpt
cd ckpt
mkdir t5-v1_1-xxl
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/text_encoder/config.json
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/text_encoder/model-00001-of-00002.safetensors
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/text_encoder/model-00002-of-00002.safetensors
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/text_encoder/model.safetensors.index.json
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/tokenizer/added_tokens.json
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/tokenizer/special_tokens_map.json
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/tokenizer/spiece.model
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/tokenizer/tokenizer_config.json

cd ../
mkdir vae
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
