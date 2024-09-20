# FPINNs

This is the official implementation of "Deep Fuzzy Physics-Informed Neural Networks for Forward and Inverse PDE Problems"
 (Neural Networks 2024) by Wenyuan Wu<sup>1</sup>, Siyuan Duan<sup>1</sup>, Yuan Sun, Yang Yu, Dong Liu and Dezhong Peng. (<sup>1</sup>denotes equal contribution, PyTorch Code)

<p align="center">
<img src="https://github.com/siyuancncd/FPINNs/blob/main/FPINN.png" width="850" height="500">
</p>

## Abstract
As a grid-independent approach for solving partial differential equations (PDEs), Physics-Informed Neural Networks (PINNs) have garnered significant attention due to their unique capability to simultaneously learn from both data and the governing physical equations. Existing PINNs methods always assume that the data is stable and reliable, but data obtained from commercial simulation software often inevitably have ambiguous and inaccurate problems. Obviously, this will have a negative impact on the use of PINNs to solve forward and inverse PDE problems. To overcome the above problems, this paper proposes a Deep Fuzzy Physics-Informed Neural Networks (FPINNs) that explores the uncertainty in data. Specifically, to capture the uncertainty behind the data, FPINNs learns fuzzy representation through the fuzzy membership function layer and fuzzy rule layer. Afterward, we use deep neural networks to learn neural representation. Subsequently, the fuzzy representation is integrated with the neural representation. Finally, the residual of the physical equation and the data error are considered as the two components of the loss function, guiding the network to optimize towards adherence to the physical laws for accurate prediction of the physical field. Extensive \rev{experiment} results show that FPINNs outperforms these comparative methods in solving forward and inverse PDE problems on four widely used datasets. The demo code will be released at [https://github.com/siyuancncd/FPINNs](https://github.com/siyuancncd/FPINNs).

## Environment Installation
```
numpy==1.26.4
torch==2.0.1
scipy==1.13.1
matplotlib==3.3.4
```
## Train Allen-Cahn:

Forward PDE Problem:
```
python Allen-Cahn_FPINNs_Forward.py
```
Inverse PDE Problem:
```
python Allen-Cahn_FPINNs_Inverse.py
```

## Citation
```
...
```

## Acknowledgement
The code is inspired by [Physics-Informed-Neural-Networks](https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks).
