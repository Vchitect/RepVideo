# RepVideo

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://vchitect.intern-ai.org.cn/' target='_blank'>Vchitect Team<sup>1</sup></a>&emsp;
</div>
<div>
<div align="center">
    <sup>1</sup>Shanghai Artificial Intelligence Laboratory&emsp;
</div>
 
 
</p>
<!-- <p align="center">
    👋 Join our <a href="https://github.com/Vchitect/RepVideo/tree/master/assets/channel/lark.jpeg" target="_blank">Lark</a> and <a href="https://discord.gg/aJAbn9sN" target="_blank">Discord</a> 
</p> -->

---

![](https://img.shields.io/badge/RepVideo-v0.1-darkcyan)
![](https://img.shields.io/github/stars/Vchitect/RepVideo)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FRepVideo&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
<!-- [![Generic badge](https://img.shields.io/badge/DEMO-Vchitect2.0_Demo-<COLOR>.svg)](https://huggingface.co/spaces/Vchitect/RepVideo) -->
[![Generic badge](https://img.shields.io/badge/Checkpoint-red.svg)](https://huggingface.co/Vchitect/Vchitect-XL-2B)




<!-- **:fire:The technical report is coming soon!**

## 🔥 Update and News
- [2024.09.14] 🔥 Inference code and [checkpoint](https://huggingface.co/Vchitect/Vchitect-XL-2B) are released.

## :astonished: Gallery

<table class="center">

<tr>

  <td><img src="assets/samples/sample_0_seed3.gif"> </td>
  <td><img src="assets/samples/sample_1_seed3.gif"> </td>
  <td><img src="assets/samples/sample_3_seed2.gif"> </td> 
</tr>


        
<tr>
  <td><img src="assets/samples/sample_4_seed1.gif"> </td>
  <td><img src="assets/samples/sample_4_seed4.gif"> </td>
  <td><img src="assets/samples/sample_5_seed4.gif"> </td>     
</tr>

<tr>
  <td><img src="assets/samples/sample_6_seed4.gif"> </td>
  <td><img src="assets/samples/sample_8_seed0.gif"> </td>
  <td><img src="assets/samples/sample_8_seed2.gif"> </td>      
</tr>

<tr>
  <td><img src="assets/samples/sample_12_seed1.gif"> </td>
  <td><img src="assets/samples/sample_13_seed3.gif"> </td>
  <td><img src="assets/samples/sample_14.gif"> </td>    
</tr>

</table> -->


## Installation

### 1. Create a conda environment and download models


  ```bash
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
  ```

## Inference

~~~bash
bash run.sh
~~~


## 🔑 License

This code is licensed under Apache-2.0. The framework is fully open for academic research and also allows free commercial usage.


## Disclaimer

We disclaim responsibility for user-generated content. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities. It is prohibited for pornographic, violent and bloody content generation, and to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.

