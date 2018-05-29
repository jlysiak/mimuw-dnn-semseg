# Semantic segmentation covolutional network

Assignment project in Deep Neural Networks  

MIMUW 2017/18  

Author: Jacek Łysiak

## Description

Command line tool for semantic segmentation with deep conv nets.
Some useful features are implemented like:

- well configurable network architecture   
- many options in config   
- overriding opts in config using CLI flags    
- validation process on demand with truth/prediction comparison    
- configurable data augmentation both in training and validation   
  - training augmentation:    
    - LR flip 
    - random non-central crops  
    - central crops   
  - valid/predict augmentation:   
    - central crops   
    - fixed non-central crops   
- selective prediction generator   
- TensorBoard logs

`trainer` is entry to CLI app.
Main part is in `semseg` directory.


