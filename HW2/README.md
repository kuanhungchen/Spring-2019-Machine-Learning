# Homework 2
* Name: 陳冠弘
* ID: 105061171

## Overview
In this homework, we need to implement a linear regression method and compute squared error for testing data.

## Implementation
### Prepare data
* We read .csv files and save as numpy array.
```python
class Regressor:
    .
    .
    .
    def load_data(self):
        """
        :return: all training data and testing data
        """
        try:
            training_data = np.loadtxt(open(self._path_to_training_data, 'r'), delimiter=',')
        except:
            raise FileNotFoundError
        try:
            testing_data = np.loadtxt(open(self._path_to_testing_data, 'r'), delimiter=',')
        except:
            raise FileNotFoundError
        return training_data, testing_data
```
### Set up statics
* Given a pair of (o1, o2), we compute the (s1, s2) from training data
```python
class Regressor:
    .
    .
    .
    def calculate_static(self, o1, o2):
        """
        set up statics from training data given o1 and o2
        :param o1: number of locations along horizontal direction
        :param o2: number of locations along vertical direction
        """
        _max = np.max(self._training_data[:, 0])
        _min = np.min(self._training_data[:, 0])
        _s1 = float((_max - _min) / (o1 - 1))
        _max = np.max(self._training_data[:, 1])
        _min = np.min(self._training_data[:, 1])
        _s2 = float((_max - _min) / (o2 - 1))
        self._static = _s1, _s2
        print("Set up statics for (o1, o2) = ({:d}, {:d})".format(o1, o2))
```
### Compute weights
* Then we compute the maximum likelihood weights from training data
```python
class Regressor:
    .
    .
    .
    def calculate_weight(self, o1, o2):
        """
        set up weight from training data given o1 and o2
        :param o1: number of locations along horizontal direction
        :param o2: number of locations along vertical direction
        """
        _PHI = np.zeros([self._training_data.shape[0], o1*o2+2])
        for i in range(_PHI.shape[0]):
            _phi = self.phi(self._training_data[i, 0:3], o1, o2)
            _PHI[i, :] = _phi
        self._weight =  np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(_PHI), _PHI)), np.transpose(_PHI)), self._training_data[:, 3])
        print("Set up weights for (o1, o2) = ({:d}, {:d})".format(o1, o2))
```
### Feature vector
* Given a new data, we compute a feature vector of it, which size will be (o1*o2+2)
* IMPORTANT: Note that the last two component are research (0 or 1) and fixed 1.
```python
class Regressor:
    .
    .
    .
    def phi(self, x, o1, o2):
        """
        use static to compute the feature vector of a new data
        :param x: a new data, x=(x1, x2, x3)
        :param o1: number of locations along horizontal direction
        :param o2: number of locations along vertical direction
        :return: feature vector, whose size = (o1 * o2 + 2)
        """
        _phi = np.ones(o1*o2+2)
        _phi[o1*o2] = x[2]
        _phi[o1*o2+1] = 1
        for i in range(1, o1+1):
            for j in range(1, o2+1):
                _inner1 = ((x[0] - self._static[0] * (i-1))**2) / (2 * (self._static[0])**2)
                _inner2 = ((x[1] - self._static[1] * (j-1))**2) / (2 * (self._static[1])**2)
                _phi[o2*(i-1) + j - 1] = np.exp(- _inner1 - _inner2)
        return _phi
```
### Predict
* Given a new data, we can compute our regression result based on the calculated maximum likelihood weights.
```python
class Regressor:
    .
    .
    .
    def predict(self, x, o1, o2):
        """
        predict a result given a new data
        :param x: a new data, x=(x1, x2, x3)
        :param o1: number of locations along horizontal direction
        :param o2: number of locations along vertical direction
        :return: prediction, 0<=y<=1
        """
        assert self._static is not None, 'Please set up statics before calculating squared error.'
        assert self._weight is not None, 'Please set up weights before calculating squared error.'
        assert self._weight.shape[0] == o1*o2+2, 'Weights size error. Please set up correct weights.'

        _pred = np.sum(self._weight[i] * self.phi(x, o1, o2)[i] for i in range(o1*o2+2))
        return _pred
```
### Calculate squared error
* Finally, we can calculate the squared error for each data in testing set. (I sum them up to obtain total squared error.)
```python
class Regressor:
    .
    .
    .
    def calculate_squared_error(self, o1, o2):
        """
        use trained weight to predict answers from testing data, then compute the
        squared error for each data
        :return: each square error, which are 100 numbers
        """
        assert self._static is not None, 'Please set up statics before calculating squared error.'
        assert self._weight is not None, 'Please set up weights before calculating squared error.'
        assert self._weight.shape[0] == o1*o2+2, 'Weights size error. Please set up correct weights.'

        squared_error = np.zeros(self._testing_data.shape[0])
        for i in range(squared_error.shape[0]):
            pred = self.predict(self._testing_data[i, 0:3], o1, o2)
            squared_error[i] = (pred - self._testing_data[i, 3]) ** 2

        return squared_error
```
### Visualize
* We can print all prediction and ground truth for testing data.
```python
class Regressor:
    .
    .
    .
    def visualize(self, o1, o2):
        """
        show prediction of testing data and also the ground truth
        :param o1: number of locations along horizontal direction
        :param o2: number of locations along vertical direction
        """
        assert self._static is not None, 'Please set up statics before calculating squared error.'
        assert self._weight is not None, 'Please set up weights before calculating squared error.'
        assert self._weight.shape[0] == o1*o2+2, 'Weights size error. Please set up correct weights.'

        print('for (o1, o2) = ({:d}, {:d})'.format(o1, o2))
        print(' GT  |  Pred ')
        print('-------------')
        for idx in range(self._testing_data.shape[0]):
            pred = self.predict(self._testing_data[idx, 0:3], o1, o2)
            print('{:.2f} | {:.4f}'.format(self._testing_data[idx, 3], pred))
        print('-------------')
```
## How to run
### Prerequisites
* Package
    * numpy
    * platform
* Python
    * python 3
### Execute
* Run  
    * in terminal (use command): ```$ python regressor.py```
    * in IDE (PyCharm, Spyder, etc.): just click the RUN button for ```regressor.py```
    * Note that the structure should be:
        ```
            .
            ├── regressor.py
            └── data --┬── Training_set.csv
                       └── Testing_set.csv
        ```
        Otherwise you need to change the paths in ```regressor.py```
        ```python
        def main():
            assert platform.python_version().split('.')[0] == '3', 'python version should be 3.X'
            _regressor = Regressor(path_to_training_data='./data/Training_set.csv', path_to_testing_data='./data/Testing_set.csv')
            .
            .
            .
        ```
    * Before predicting/calculating error/visualizing, you should set statics and compute weights first. (Don't worry, there will be an assertion if you forget.) 
    * To use different (o1, o2), please modify line 132, 134 in ```regressor.py```
    * Note that normally we use same (o1, o2) for setting statics and predicting/calculating error/visualizing, but sometimes using different (o1, o2) results in better performance. (For example, I found that using (2, 2) to set statics and (6, 3) for predicting can have a nice performance.)
    * Note that the pair of (o1, o2) for computing weights and for predicting/calculating error/visualizing should be the same. (Don't worry, there will be an assertion if you use different (o1, o2).)
    * To compute total squared error, please uncomment line 136 in ```regressor.py```
    * To try an example, please uncomment line 138, 139, 140 in ```regressor.py``` (You can use different example data by modifying line 138.)
    * To see the visulization, please uncomment line 142 in ```regressor.py```
    
* Ablation study
    * in terminal (use command): ```$ python ablation.py```
    * in IDE (PyCharm, Spyder, etc.): just click the RUN button for ```ablation.py```
    * Note that the structure should be:
        ```
            .
            ├── ablation.py
            └── data --┬── Training_set.csv
                       └── Testing_set.csv
        ```
        Otherwise you need to change the paths in ```ablation.py```
        ```python
        def main():
          _regressor = Regressor(path_to_training_data='./data/Training_set.csv', path_to_testing_data='./data/Testing_set.csv')
          .
          .
          .
        ```
    * Note that here I use same (o1, o2) for setting statics, computing weights, and predicting.
    * To see the visulization, please uncomment line 13, 14 in ```ablation.py```

## Result
* Using SAME (o1, o2) for setting statics and calculating total squared error.
    ```python
    Set up statics for (o1, o2) = (2, 2)
    Set up weights for (o1, o2) = (2, 2)
    for (o1, o2) = (2, 2), squared error: 0.5870
    ``` 
    ```python
    Set up statics for (o1, o2) = (2, 3)
    Set up weights for (o1, o2) = (2, 3)
    for (o1, o2) = (2, 3), squared error: 0.7808
    ```
    ```python
    Set up statics for (o1, o2) = (3, 2)
    Set up weights for (o1, o2) = (3, 2)
    for (o1, o2) = (3, 2), squared error: 0.9506
    ```
    ```python
    Set up statics for (o1, o2) = (3, 3)
    Set up weights for (o1, o2) = (3, 3)
    for (o1, o2) = (3, 3), squared error: 1.0029
    ```
    
* Using DIFFERENT (o1, o2) for setting and calculating total squared error.
    ```python
    Set up statics for (o1, o2) = (2, 2)
    Set up weights for (o1, o2) = (3, 3)
    for (o1, o2) = (3, 3), squared error: 0.4908
    ```
    ```python
    Set up statics for (o1, o2) = (2, 2)
    Set up weights for (o1, o2) = (5, 3)
    for (o1, o2) = (5, 3), squared error: 0.4578
    ```
    ```python
    Set up statics for (o1, o2) = (2, 2)
    Set up weights for (o1, o2) = (6, 3)
    for (o1, o2) = (6, 3), squared error: 0.4561
    ```

* Prediction examples
    ```python
    Set up statics for (o1, o2) = (2, 2)
    Set up weights for (o1, o2) = (2, 2)
    Feature = [321, 112, 1], ground truth = [0.77]
    Prediction = [0.8181]
    ```
    ```python
    Set up statics for (o1, o2) = (2, 2)
    Set up weights for (o1, o2) = (6, 3)
    Feature = [321, 112, 1], ground truth = [0.77]
    Prediction = [0.8044]
    ```

* Visualization examples
    * SAME (o1, o2)
        ```python
        Set up statics for (o1, o2) = (2, 2)
        Set up weights for (o1, o2) = (2, 2)
        for (o1, o2) = (2, 2)
         GT  |  Pred 
        -------------
        0.63 | 0.5476
        0.66 | 0.6982
        0.78 | 0.8193
        0.91 | 0.8434
        0.62 | 0.6963
        0.52 | 0.5171
          .  |    .
          .  |    .
          .  |    .
        0.96 | 0.8497
        0.93 | 0.8471
        0.73 | 0.6603
        0.84 | 0.7764
        -------------
        ```
    * DIFFERENT (o1, o2)
        ```python
        Set up statics for (o1, o2) = (2, 2)
        Set up weights for (o1, o2) = (6, 3)
        for (o1, o2) = (6, 3)
         GT  |  Pred 
        -------------
        0.63 | 0.5641
        0.66 | 0.6769
        0.78 | 0.7786
        0.91 | 0.8891
        0.62 | 0.6681
        0.52 | 0.5336
          .  |    .
          .  |    .
          .  |    .
        0.96 | 0.9352
        0.93 | 0.9152
        0.73 | 0.6511
        0.84 | 0.8028
        -------------
        ```
