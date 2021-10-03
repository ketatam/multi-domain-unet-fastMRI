## Multi Domain U-Net Model for MRI Reconstruction

Welcome to the repository of my Bachelor's thesis at the Department of Electrical and Computer Engineering at [TUM](https://www.tum.de/en/). 
My thesis was about "Data Standardization, Multi-Domain Learning and GRAPPA preprocessing for Improved MRI".
To read a more detailed discussion see my [thesis]() and [final presentation]().

This directory contains implementations of U-Net and multi-domain-U-Net for MRI reconstruction in PyTorch. It also contains 
implementations of the different methods I used in my work, so that the reported results are easily reproducible.

To visit the main page of the fastMRI challenge, please go [here](https://fastmri.org/).
## Dependencies and Installation

I have tested this code using:

* Ubuntu 16.04.6
* Python 3.6.9
* CUDA 10.2

You can find the full list of Python packages needed to run the code in the `requirements.txt` file.
To install them, run

```bash
pip install -r requirements.txt
```

## Repository Structure
This repository is structured as follows:
* The directory `tutorials` contains three Jupyter notebooks that illustrate how to deal with the data and how to use the implemented methods.
This is the best place to start.
  
* The other directory `data`, `common` and `models` contain the actual implementations of the model training pipeline. See
the respective folders for a more detailed explanation.

