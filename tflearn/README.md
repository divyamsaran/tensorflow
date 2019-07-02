# TFLearn
I create a simple CNN to classify MNIST dataset using TFLearn. I use two convolution layers with small sized kernels (3 X 3), RELU activations, L2 regularizer, followed by max pooling. I then have two fully connected layers, with the first having a dropout with drop probability of 20% and a RELU activation, and the second having a softmax activation for the ten classes. I use cross entropy loss and Adam optimizer. 

![Model](results/tflearn_graph.png)

The model achieves a test accuracy of 98.14% after 10 epochs.

![Accuracy](results/tflearn.png)

I also show the easy integration of TFLearn with tensorboard, here are a few generated visualizations:

![Validation Accuracy](results/tflearn_validation_accuracy.png)

![Accuracy over epochs](results/tflearn_accuracy.png)

![Optimizer](results/tflearn_adam.png)

![Loss over epochs](results/tflearn_loss.png)