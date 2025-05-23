# Handwritten Digit Classification with Custom RNN (on MNIST)

This project implements a custom Recurrent Neural Network (RNN) from scratch in PyTorch to classify handwritten digits from the MNIST dataset. The model is manually built using weight matrices and activation functions, with no use of PyTorch’s built-in `nn.RNN` or other higher-level modules.

## Features

- Custom RNN implementation using PyTorch low-level operations.
- Support for both SGD and Adam optimizers.
- Command-line arguments for easy hyperparameter tuning.
- Training log includes timestamp, accuracy, loss, and learning rate.

## Dependencies

Install the required packages with:

```bash
pip3 install torch torchvision numpy
```

## Usage
Train the model
``` bash
python main.py
```
Customize training parameters
```bash
python main.py -optim adam -lr 0.001 -bs 128 -hid_dim 256 -ep 20
```
## Model Architecture
- Input: MNIST 28x28 grayscale images.
- RNN Input Format: Each row (28 pixels) is treated as a time step.
- Manual RNN Cell:
  - $h_t = tanh$($x_t$  @ $W_xh$ + $h_{t-1}$ @ $W_hh$ + $b_h$)
  - Final prediction: $y$ = $h_T$ @ $W_yh$ + $b_y$
- Loss Function: CrossEntropyLoss
- Evaluation: Accuracy on test set per epoch
## Example Output
```bash
MNIST Train Dataset is 60000 samples, Test Dataset is 10000 samples.
Before Training, The Accuracy of Test Loader is 0.0764.
Epoch: [ 0/50] | Train Loss: 0.5393  | Accuracy: 90.50 % | LR: 0.00950 | Time: 28.11 s | Now: 2025-05-23 14:38:41.
Epoch: [ 1/50] | Train Loss: 0.2939  | Accuracy: 92.71 % | LR: 0.00903 | Time: 28.45 s | Now: 2025-05-23 14:39:09.
Epoch: [ 2/50] | Train Loss: 0.2195  | Accuracy: 94.69 % | LR: 0.00857 | Time: 30.66 s | Now: 2025-05-23 14:39:40.
...
```

## References

- [CS231n: Convolutional Neural Networks for Visual Recognition – Stanford University](http://cs231n.stanford.edu/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ChatGPT (by OpenAI)](https://chat.openai.com/) — Used for debugging and architecture refinement
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
