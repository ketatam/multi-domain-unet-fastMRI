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
