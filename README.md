# pytorch-pwc

This is a refactored fork of [this implementation](https://github.com/sniklaus/pytorch-pwc) of PWC-Net. The code has been restructured to be easier to use and work on the latest versions of libraries.
Works on PyTorch 1.0 and Python 3.7.

## Dependencies

 * pytorch >=1.0
 * python >=3.7
 * cupy
 * numpy

## License
As stated in the <a href="https://github.com/NVlabs/PWC-Net#license">licensing terms</a> of the authors of the paper, the models are free for non-commercial share-alike purpose. Please make sure to further consult their licensing terms.

## References
```
[1]  @inproceedings{Sun_CVPR_2018,
         author = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
         title = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```

```
[2]  @misc{pytorch-pwc,
         author = {Simon Niklaus},
         title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
    }
```

