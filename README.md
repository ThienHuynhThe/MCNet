# MCNet

Abstract - This letter proposes a cost-efficient convolutional neural network (CNN) for robust automatic modulation classification (AMC) deployed for cognitive radio services of modern communication systems. The network architecture is designed with several specific convolutional blocks to concurrently learn the spatiotemporal signal correlations via different asymmetric convolution kernels. Additionally, these blocks are associated with skip connections to preserve more initially residual information at multi-scale feature maps and prevent the vanishing gradient problem. In the experiments, MCNet reaches the overall 24-modulation classification rate of 93.59% at 20 dB SNR on the well-known DeepSig dataset.

![Image of Architecture](https://github.com/ThienHuynhThe/MCNet/blob/master/architecture-eps-converted-to.pdf)

T. Huynh-The, C. Hua, Q. Pham and D. Kim, "MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification," in IEEE Communications Letters, vol. 24, no. 4, pp. 811-815, April 2020, doi: 10.1109/LCOMM.2020.2968030.

@ARTICLE{mcnet2020CommLett,
  author={T. {Huynh-The} and C. {Hua} and Q. {Pham} and D. {Kim}},
  journal={IEEE Communications Letters}, 
  title={MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification}, 
  year={2020},
  volume={24},
  number={4},
  pages={811-815},}

We provide the MATLAB code of automatic modulation classification of this paper.

