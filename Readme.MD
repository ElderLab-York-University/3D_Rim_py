# 3D rim from contours (python version)

## overview
This project provides a method to analyze the normal/apparent curvature on 2D occluding contours. 

## Code:

code was written in python 3.10 and design to running on the [Thingi10k](https://arxiv.org/abs/1605.04797) dataset

To install required packages for the project run
`pip install -r requirements.txt` in the root project
directory.

## how to use

put your models in .\Thingi10kData folder. or you can change desired 3d model path in `config.ini`

run `data_set_sampling.py` will generate npz file that contain 2d contours, 3d object path, view angle, normal and apparent curvature.
in `.\npzOutPath` you can change  desired output path in `config.ini`

run `npz_paser.py` will output a tuple with 7 dict, each dict has view angle as key and a list of data as value.