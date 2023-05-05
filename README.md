Table recognition 🧠
====================

This repository contains a code for training and using a table recognition
model. 

Video 🎞️
--------
<p align="center">
<a href="https://youtu.be/x_FXpi2wP14"><img src="http://img.youtube.com/vi/x_FXpi2wP14/0.jpg"></a>
</p>

Requirements
------------
- The tested was tested using ```python v3.7.12```
- To use the code please install the requirements first
  ```bash
    # Installation of CPU packages
    virtualenv .venv && source .venv/bin/activate
    pip install -r requirements-cpu.txt
  ```
- To use or train the model using GPU it is necessary to install packages
  listed in requirements-gpu.txt (to use this packages you have to have 
  ```cuda 11.2``` installed on your system)
  ```bash
    # Installation of GPU packages
    virtualenv .venv && source .venv/bin/activate
    pip install -r requirements-gpu.txt
  ```
