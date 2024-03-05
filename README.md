# Assignment1 (Scratch Implementing MULTI-LAYER PERCEPTRON )
![image](https://github.com/mohammadfaizan76692/Deep-LearningCSE-641/assets/77170022/e34e62bd-a636-43fb-9495-a281d3993705)

### Question 1 
    1) Implementing DataLoader and Dataset from Scratch for MNIST Dataset
    2) Comparing the Performance of DataLaoder from Scratch and the one provided by Pytorch with batch sizes [128, 256, 512, 1024].

### Question 2: Implementing MLP using Pytorch Module.
    1) Model Architecture : input Neurons = 728 (28*28) Pixels , Hidden Layers = [32,32,32,32] , Output Layers = 10 (num of classes)
    2) Activation function: ReLu (Rectified Linear Unit)
    3) Optimizer  = SGD(Stochastic Gradient Descent)
    4) Loss Function = Cross Entropy Loss

### Question 3: Do Everything From Scratch and Implement BackWard Propagation Also from Scratch.

### Question 4: Do Question 2 and Question 3 Using Activation as Sigmoid and Compare Results with ReLu.
    
___
# Assignment2 (Implement Architecture Related to CNN (Convolutional Neural Networks) 
### Question1: ResNet's Architecture
![image](https://github.com/mohammadfaizan76692/Deep-LearningCSE-641/assets/77170022/d5971792-12ae-482d-94fd-5618269aef30)


    1)  Dataset  = CIFAR-10 and SpeechCommand V0.02.
    2) Creating this Resnet Block And using thing Blocks for Creating Architecture Consist 4 of these Blocks
    3)  Do 2D convolution on Image Dataset and 1D convolution on Audio Dataset
    4) Optimizer = Adam  and epochs =64

### Question2: Modified  VGG NET Architecture 
![image](https://github.com/mohammadfaizan76692/Deep-LearningCSE-641/assets/77170022/e57ce1a6-bcdf-4db9-94b5-a9be0d87b990)

    1)  Dataset  = CIFAR-10 and SpeechCommand V0.02.
    2) After each pooling layer, the number of channels is reduced by 35%, and the kernel size is increased by 25%             (with ceil rounding for float calculations).
    3)  Do 2D convolution on Image Dataset and 1D convolution on Audio Dataset
    4) Optimizer = Adam  and epochs =64

### Question3: Modified Inception Module
![image](https://github.com/mohammadfaizan76692/Deep-LearningCSE-641/assets/77170022/c02bab07-f388-4ea9-aff5-c94684edce0c)

    1)  Dataset  = CIFAR-10 and SpeechCommand V0.02.
    2) Construct a modified inception network comprising 4 such blocks
    3) Do 2D convolution on Image Dataset and 1D convolution on Audio Dataset
    4) Optimizer = Adam  and epochs =64

### Question4: Combination of Inception and ResNet blocks Architecture
    1) Architecture
        (a) Input Layer
        (b) Residual Block × 2
        (c) Inception Block × 2
        (d) Residual Block × 1
        (e) Inception Block × 1
        (f) Residual Block × 1
        (g) Inception Block × 1
        (h) Residual Block × 1
        (i) Inception Block × 1
        (j) Classification Network
        1)  Dataset  = CIFAR-10 and SpeechCommand V0.02.
    2) Dataset  = CIFAR-10 and SpeechCommand V0.02.
    3) Do 2D convolution on Image Dataset and 1D convolution on Audio Dataset
    4) Optimizer = Adam  and epochs =64
