## Introduction
This web UI is for simple computational calculations with [Psi4](https://psicode.org/):

***Single-Point Calculation***
![Screenshot 1](/images/webui1.png)
![Screenshot 2](/images/webui2.png)

***Geometry Optimization***
![Screenshot 3](/images/webui3.png)
![Screenshot 4](/images/webui4.png)

***Frequency Analysis***
![Screenshot 5](/images/webui5.png)
![Screenshot 6](/images/webui6.png)

## Installation
You will need [Anaconda](https://www.anaconda.com/download) for this app.
- Clone this repo:
`git clone https://github.com/phatdatnguyen/psi4-webui`
- Open terminal:
`cd psi4-webui`
- Create and activate Anaconda environment:
`conda create -p ./psi4-env`
`conda activate ./psi4-env`
- Install packages:
`conda install conda-forge::rdkit`
`conda install psi4 -c conda-forge/label/libint_dev -c conda-forge`
`pip install gradio`
`pip install gradio_molecule2d`
`pip install gradio_molecule3d`

## Start web UI
To start the web UI:
`conda activate ./psi4-env`
`set PYTHONUTF8=1`
`python webui.py`