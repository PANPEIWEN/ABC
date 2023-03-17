# ABC: Attention with Bilinear Correlation for Infrared Small Target Detection
Infrared small target detection (ISTD) has a wide range of applications in early warning, rescue, and guidance. However, CNN based deep learning methods are not effective at segmenting infrared small target (IRST) that it lack of clear contour and texture features, and transformer based methods also struggle to achieve significant results due to the absence of convolution induction bias. To address these issues, we propose a new model called attention with bilinear correlation (ABC), which is based on the transformer architecture and includes a convolution linear fusion transformer (CLFT) module with a novel attention mechanism for feature extraction and fusion, which effectively enhances target features and suppresses noise. Additionally, our model includes a u-shaped convolution-dilated convolution (UCDC) module located deeper layers of the network, which takes advantage of the smaller resolution of deeper features to obtain finer semantic information. Experimental results on public datasets demonstrate that our approach achieves state-of-the-art performance.

## Installation
```angular2html
pip install -U openmim
mim install mmcv-full==1.7.0
mim install mmdet==2.25.0
mim install mmsegmentation==0.28.0
```
You may also need to install other packages, if you encounter a package missing error, you just need to install it using the pip command.
## Dataset Preparation
### File Structure
```angular2html
|- datasets
   |- NUAA
      |-trainval
        |-images
          |-Misc_1.png
          ......
        |-masks
          |-Misc_1.png
          ......
      |-test
        |-images
          |-Misc_50.png
          ......
        |-masks
          |-Misc_50.png
          ......
   |-IRSTD1k
   |-NUDT
   |-SIRSTAUG

```
Please make sure that the path of your data set is consistent with the `data_root` in `configs/_base_/datasets/dataset_name.py`
### Datasets Link
https://drive.google.com/drive/folders/1RGpVHccGb8B4_spX_RZPEMW9pyeXOIaC?usp=sharing

## Training
### Single GPU Training


