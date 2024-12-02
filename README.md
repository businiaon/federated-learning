python main.py --dataset mnist --model_name MLP --epochs 20 --batch_size 64 --learning_rate 0.01
python experiments.py --dataset mnist --model_name MLP --num_rounds 100 --num_clients_per_round 10 --train_mlp True --train_cnn False

# Federated Learning with Logistic Regression, MLP, and CNN 

This repository provides an implementation of federated learning using Logistic Regression, MLP, and CNN models. The experiments are run on the MNIST and CIFAR-10 datasets with both IID (Independent and Identically Distributed) and non-IID data distribution settings.

The goal is to demonstrate the effectiveness of federated learning for training models across multiple clients with decentralized data.

## Data
### 1. MNIST (Modified National Institute of Standards and Technology)
- Overview: MNIST is a classic dataset used for training and testing image classification models. It consists of 28x28 grayscale images of handwritten digits (0–9), and is often used as a benchmark for evaluating machine learning algorithms. The dataset is widely used in the machine learning community for tasks like supervised learning and neural network training.
- Dataset Characteristics:
  - Number of Images: 60,000 training images and 10,000 test images.
  - Image Size: 28x28 pixels.
  - Image Type: Grayscale (single channel, pixel values between 0 and 255, where 0 is black and 255 is white).
  - Number of Classes: 10 classes (digits 0 through 9).
- Input Features:
  - Shape: Each image is a 28x28 matrix (784 features if flattened into a vector).
  - Pixel Values: The pixel values range from 0 (black) to 255 (white).
  - When used for training, the image pixel values are usually normalized to the range [0, 1] by dividing each pixel value by 255.

- Output Labels:
    - Classes: 10 classes corresponding to the digits 0–9.
    - Label Encoding: Each image has a corresponding label, where:
      - 0 represents the digit "0"
      - 1 represents the digit "1"
      - and so on until 9 for the digit "9".
- Example: For an image containing the handwritten digit "5", the input will be a 28x28 grayscale matrix of pixel values, and the output will be the label 5.

### 2. CIFAR-10 (Canadian Institute for Advanced Research)
- Overview: CIFAR-10 is another popular dataset in the field of machine learning. It consists of 60,000 32x32 color images across 10 different classes. The dataset is commonly used to evaluate the performance of models on real-world images and is much more complex compared to MNIST due to its color images and greater variety of object categories.

- Dataset Characteristics:
  - Number of Images: 60,000 total images.
  - 50,000 training images.
  - 10,000 test images.
  - Image Size: 32x32 pixels.
  - Image Type: Color (RGB, 3 channels: Red, Green, Blue).
  - Number of Classes: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- Input Features:
  - Shape: Each image is a 32x32x3 matrix (3 channels corresponding to RGB, each with 32x32 pixels).
  - Pixel Values: Each pixel value ranges from 0 to 255 for each channel (RGB).
  - The images are usually normalized by scaling the pixel values to the range [0, 1] for neural network training.

- Output Labels:
  - Classes: 10 classes, corresponding to the following object categories: [Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck]
  - Label Encoding: Each image is associated with one of these 10 classes, with labels from 0 to 9: { 0: Airplane, 1: Automobile, 2: Bird, 3: Cat, 4: Deer, 5: Dog, 6: Frog, 7: Horse, 8: Ship, 9: Truck}
- Example: For an image containing a "bird", the input will be a 32x32x3 matrix (3 channels: R, G, B), and the output will be the label 2.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python fl_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python fl_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```


## Configuration Parameters
The configuration parameters for the experiments are set in the ```config.py``` file. Some important parameters are:

Basic Configuration:
* ```--seed:``` Random seed for reproducibility. Default: 42
* ```--device:``` Set to cpu for CPU or cuda for GPU. Default: cpu
* ```--batch_size:``` Batch size for training. Default: 64
* ```--learning_rate:``` Learning rate for the optimizer. Default: 0.01
* ```--epochs:``` Number of training epochs. Default: 20
Dataset and Model Configuration:
* ```--dataset:``` Dataset to use for training. Options: mnist, cifar10. Default: mnist
* ```--model_name:``` Model to train. Options: LogisticRegression, MLP, CNN. Default: MLP
* ```--input_dim:``` Input dimension for the model (e.g., 28*28 for MNIST). Default: 28*28
* ```--hidden_dim:``` Hidden dimension for MLP model. Default: 128
* ```--num_classes:``` Number of output classes (10 for MNIST and CIFAR-10). Default: 10
Federated Learning Configuration:
* ```--num_clients_per_round:``` Number of clients per round in federated learning. Default: 10
* ```--num_local_epochs:``` Number of local epochs per client. Default: 5
* ```--num_rounds:``` Number of federated learning rounds. Default: 100
* ```--iid:``` Whether the data is IID (True) or non-IID (False). Default: True
Save Path:
* ```--save_path:``` Path to save intermediate results such as model checkpoints and logs. Default: ./save

## Results on MNIST

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />

```Table 1:``` Test accuracy after training for 100(IID) and 500(non-IID) global epochs with:

| Model |    IID   | Non-IID|
| ----- | -----    |----            |
|  LR  |  92.38%  |     90%     |
|  MLP  |  97.8%  |     96%     |
|  CNN  |  99%  |     97.8%     |


## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
