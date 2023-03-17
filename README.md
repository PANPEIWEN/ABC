# ABC
ABC: Attention with Bilinear Correlation for Infrared Small Target Detection

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


