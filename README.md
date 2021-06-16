## Multi Domain U-Net Model for MRI Reconstruction

This directory contains implementations of U-Net and multi-domain-U-Net for MRI reconstruction in PyTorch.

## Dependencies and Installation

We have tested this code using:

* Ubuntu 16.04.6
* Python 3.6.9
* CUDA 10.2

You can find the full list of Python packages needed to run the code in the `requirements.txt` file.
To install them, run

```bash
pip install -r requirements.txt
```

## U-Net
To start training the model, run:
```bash
python models/unet/train_unet.py --mode train --challenge CHALLENGE --data-path DATA --exp unet --mask-type MASK_TYPE --standardize --apply-grappa
```
where `CHALLENGE` is either `singlecoil` or `multicoil`. And `MASK_TYPE` is either `random` (for knee)
or `equispaced` (for brain). Also, add `--standardize` and `--apply-grappa`, if you want to use multi-channel data standardization and GRAPPA preprocessing, respectively. Training logs and checkpoints are saved in `experiments/unet` directory. 

To run the model on test data:
```bash
python models/unet/train_unet.py --mode test --challenge CHALLENGE --data-path DATA --exp unet --out-dir reconstructions --checkpoint MODEL --standardize --apply-grappa
```
where `MODEL` is the path to the model checkpoint from `experiments/unet/version_0/checkpoints/`.

The outputs will be saved to `reconstructions` directory which can be uploaded for submission.

# Multi Domain U-Net
To start training the model, run:
```bash
python models/unet/train_MD_unet.py --mode train --challenge CHALLENGE --data-path DATA --exp unet --mask-type MASK_TYPE --standardize --apply-grappa
```
where `CHALLENGE` is either `singlecoil` or `multicoil`. And `MASK_TYPE` is either `random` (for knee)
or `equispaced` (for brain). Also, add `--standardize` and `--apply-grappa`, if you want to use multi-channel data standardization and GRAPPA preprocessing, respectively. Training logs and checkpoints are saved in `experiments_MD_unet/unet` directory. 

To run the model on test data:
```bash
python models/unet/train_MD_unet.py --mode test --challenge CHALLENGE --data-path DATA --exp unet --out-dir reconstructions --checkpoint MODEL --standardize --apply-grappa
```
where `MODEL` is the path to the model checkpoint from `experiments/unet/version_0/checkpoints/`.

The outputs will be saved to `reconstructions` directory which can be uploaded for submission.
