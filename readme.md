
# mnistevol (v1.0)
## Description 
* The objective is to generate a number of adversarial examples of the number "2" from the MNIST database to fool a MNIST classifier to appear as "6".
  
## Setting Up

### Dependencies
This library uses `TensorFlow`, `numpy`, and `matplotlib`. Matploblib is used for plotting images after the adversarial samples have been generated.

##  Discoveries
* The MNIST classifier was built through following the tutorial on tensorflow website. The MNIST classifer reached accuracy of around 99.2% on the MNIST test samples.
* Some research was done in search for a way to generate adversarial images reliably. One method proposed by Goodfellow(2014) was a method called FGSM(fast gradient sign method). I tried implementing this method with my rudimentary understanding of it, but the results were not that great since only a few samples of 2 were able to be misclassified. (perhaps it is more difficult for numbers)
* A simple genetic algorithm was built to add random mutations and evolve through maximizing probability of "6".

## Sample Images

## Resources
* https://www.tensorflow.org/versions/master/tutorials/layers/
* https://www.researchgate.net/publication/283163432_HCNN_A_Neural_Network_Model_for_Combining_Local_and_Global_Features_Towards_Human-Like_Classification
* https://arxiv.org/abs/1412.6572
