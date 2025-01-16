# RepVideo: Rethinking Cross-Layer Representation for Video Generation

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div class="is-size-5 publication-authors">
              <!-- Paper authors -->
              <span class="author-block">
                <a href="https://chenyangsi.top/" target="_blank">Chenyang Si</a><sup>1†</sup>,</span>
                <span class="author-block">
                  <a href="https://scholar.google.com/citations?user=ORlELG8AAAAJ" target="_blank">Weichen Fan</a><sup>1†</sup>,</span>
                  <span class="author-block">
                    <a href="https://scholar.google.com/citations?user=FkkaUgwAAAAJ&hl=en" target="_blank">Zhengyao Lv</a><sup>2</sup>,</span>
                  <span class="author-block">
                  <a href="https://ziqihuangg.github.io/" target="_blank">Ziqi Huang</a><sup>1</sup>,</span>
                  <span class="author-block">
                  <a href="https://mmlab.siat.ac.cn/yuqiao" target="_blank">Yu Qiao</a><sup>2</sup>,</span>
                  <span class="author-block">
                    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>1✉</sup>
                  </span>
                  </div>
<div class="is-size-5 publication-authors">
                    <span class="author-block">S-Lab, Nanyang Technological University<sup>1</sup> &nbsp;&nbsp;&nbsp;&nbsp; Shanghai Artificial Intelligence Laboratory <sup>2</sup> </span>
                    <span class="eql-cntrb"><small><br><sup>†</sup>Equal contribution.&nbsp;&nbsp;&nbsp;&nbsp;<sup>✉</sup>Corresponding Author.</small></span>
                  </div>

</p>
<!-- <p align="center">
    👋 Join our <a href="https://github.com/Vchitect/RepVideo/tree/master/assets/channel/lark.jpeg" target="_blank">Lark</a> and <a href="https://discord.gg/aJAbn9sN" target="_blank">Discord</a> 
</p> -->

---

![](https://img.shields.io/badge/RepVideo-v0.1-darkcyan)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FRepVideo&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Generic badge](https://img.shields.io/badge/Checkpoint-red.svg)](https://huggingface.co/Vchitect/RepVideo)




<!-- **:fire:The technical report is coming soon!**

## 🔥 Update and News
- [2024.09.14] 🔥 Inference code and [checkpoint](https://huggingface.co/Vchitect/Vchitect-XL-2B) are released.

## :astonished: Gallery

<table class="center">
<tr>

  <td><img src="assets/1.gif"> </td>
  <td><img src="assets/2.gif"> </td>
  <td><img src="assets/3.gif"> </td> 
</tr>


<tr>
  <td><img src="assets/4.gif"> </td>
  <td><img src="assets/5.gif"> </td>
  <td><img src="assets/6.gif"> </td>     
</tr>

<tr>
  <td><img src="assets/7.gif"> </td>
  <td><img src="assets/8.gif"> </td>
  <td><img src="assets/9.gif"> </td>      
</tr>

<tr>
  <td><img src="assets/10.gif"> </td>
  <td><img src="assets/11.gif"> </td>
  <td><img src="assets/12.gif"> </td>    
</tr>

</table> 


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
cd sat
bash run.sh
~~~


## 🔑 License

This code is licensed under Apache-2.0. The framework is fully open for academic research and also allows free commercial usage.


## Disclaimer

We disclaim responsibility for user-generated content. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities. It is prohibited for pornographic, violent and bloody content generation, and to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.

