# Homework 1
* Name: 陳冠弘
* ID: 105061171

## Overview
In this homework, we need to implement the maximum a posteriori probability of a classifier and compute error rate on testing data.

## Implementation
### Prepare data
* We convert .mat files to numpy array.
```python
class IrisClassifier:
    .
    .
    .
    def prepare_training_data(self):
        # load training data from .csv and convert to numpy array
        training_data = scipy.io.loadmat(self.path_to_training_set)

        training_sepal_length = np.asarray(training_data[list(training_data.keys())[3]][0][:])
        training_sepal_width = np.asarray(training_data[list(training_data.keys())[4]][0][:])
        training_petal_length = np.asarray(training_data[list(training_data.keys())[5]][0][:])
        training_petal_width = np.asarray(training_data[list(training_data.keys())[6]][0][:])
        training_label = np.asarray(training_data[list(training_data.keys())[7]][0][:])

        # calculate prior prob. for each class on training set
        prior_prob = [0, 0, 0]  # Iris-setosa, Iris-versicolor, Iris-virginica
        for label in training_label:
            prior_prob[int(label)] += 1
        total_num_of_training_set = prior_prob[0] + prior_prob[1] + prior_prob[2]
        prior_prob = [prior_prob[i] / total_num_of_training_set for i in range(3)]

        return total_num_of_training_set, prior_prob, training_sepal_length, training_sepal_width, training_petal_length, training_petal_width, training_label

    def prepare_testing_data(self):
        # load testing data from .csv and convert to numpy array
        testing_data = scipy.io.loadmat(self.path_to_testing_set)

        testing_sepal_length = np.asarray(testing_data[list(testing_data.keys())[3]][0][:])
        testing_sepal_width = np.asarray(testing_data[list(testing_data.keys())[4]][0][:])
        testing_petal_length = np.asarray(testing_data[list(testing_data.keys())[5]][0][:])
        testing_petal_width = np.asarray(testing_data[list(testing_data.keys())[6]][0][:])
        testing_label = np.asarray(testing_data[list(testing_data.keys())[7]][0][:])

        return testing_sepal_length, testing_sepal_width, testing_petal_length, testing_petal_width, testing_label
```
### Compute statics
* We compute mean and standard deviation for sepal length, sepal width, petal length and petal width for each classes.
```python
class IrisClassifier:
    .
    .
    .
    def calculate_statics(self):
        # calculate mean and standard deviation for each class, each feature
        sepal_length_mean = [0, 0, 0]  # Iris-setosa, Iris-versicolor, Iris-virginica
        for idx, sepal_length in enumerate(self.training_sepal_length):
            sepal_length_mean[int(self.training_label[idx])] += sepal_length
        sepal_length_mean = [sepal_length_mean[i] / (self.total_num_of_training_set*self.prior_prob[i]) for i in range(3)]

        sepal_length_stdev = [0, 0, 0]  # Iris-setosa, Iris-versicolor, Iris-virginica
        for idx, sepal_length in enumerate(self.training_sepal_length):
            sepal_length_stdev[int(self.training_label[idx])] += (sepal_length - sepal_length_mean[int(self.training_label[idx])]) ** 2
        sepal_length_stdev = [sepal_length_stdev[i] / (self.total_num_of_training_set*self.prior_prob[i]) for i in range(3)]
        sepal_length_stdev = [sepal_length_stdev[i] ** 0.5 for i in range(3)]
        .
        .
        .
        for _class in range(3):
                statics[_class][0].append(sepal_length_mean[_class])
                statics[_class][0].append(sepal_length_stdev[_class])
                statics[_class][1].append(sepal_width_mean[_class])
                statics[_class][1].append(sepal_width_stdev[_class])
                statics[_class][2].append(petal_length_mean[_class])
                statics[_class][2].append(petal_length_stdev[_class])
                statics[_class][3].append(petal_width_mean[_class])
                statics[_class][3].append(petal_width_stdev[_class])
        return statics
```
### Joint probability
* Given a new data, we firstly compute likelihood for each class, then we multiply each likelihood with corresponding prior probability to get joint probabilities.
* IMPORTANT: Note that if we only use two features (petal length, petal width) to compute likelihood, the prediction accuracy will be slightly improved. I think it's because the petal size is highly related to classes, but sepal size is not. (I found this after visualizing distributions of training and testing data.)
```python
class IrisClassifier:
    .
    .
    .
    def calculate_joint_prob(self, input_feature):
        # for a new data, calculate corresponding joint prob. for each class
        joint_probs = [0, 0, 0]  # Iris-setosa, Iris-versicolor, Iris-virginica
        for _class in range(3):
            likelihood = 1
            if self.use_petal_only:
                total_features = 2
                for _feature in range(-total_features, -1):
                    feature = input_feature[_feature]
                    mean = self.statics[_class][_feature][0]
                    stdev = self.statics[_class][_feature][1]
                    normal_prob = self.normal_pdf(feature, mean, stdev)
                    likelihood *= normal_prob
            else:
                total_features = 4
                for _feature in range(total_features):
                    feature = input_feature[_feature]
                    mean = self.statics[_class][_feature][0]
                    stdev = self.statics[_class][_feature][1]
                    normal_prob = self.normal_pdf(feature, mean, stdev)
                    likelihood *= normal_prob

            _prior_prob = self.prior_prob[_class]
            joint_probs[_class] = _prior_prob * likelihood
        return joint_probs
```
### Posterior probability
* After we have joint probabilities, we can compute posterior prob. by deviding marginal prob.(sum of joint prob.).
```python
class IrisClassifier:
    .
    .
    .
    def calculate_posterior_probs(self, input_feature):
        # for a new data, calculate corresponding posterior prob. for each class
        posterior_probs = [0, 0, 0]
        joint_probabilities = self.calculate_joint_prob(input_feature)
        marginal_prob = self.marginal_pdf(joint_probabilities)
        for idx, _joint_prob in enumerate(joint_probabilities):
            posterior_probs[idx] = _joint_prob / marginal_prob
        return posterior_probs
```
### Predict MAP
* For each new data, we choose Maximum A Posterior prob. as our prediction.
```python
class IrisClassifier:
    .
    .
    .
    def predict(self, input_feature):
        # pick maximum posterior prob. (i.e. MAP) as prediction
        posterior_probs = self.calculate_posterior_probs(input_feature)
        map_prob = max(posterior_probs[:])
        map_idx = posterior_probs.index(map_prob)
        return map_prob, map_idx
```
### Compute accuracy
* Finally, we can count how many data we predict correctly and then calculate the accuracy.
```python
class IrisClassifier:
    .
    .
    .
    def calculate_accuracy(self):
        # calculate prediction accuracy
        hit = 0
        for idx in range(self.testing_label.shape[0]):
            _, _pred = self.predict([self.testing_sepal_length[idx], self.testing_sepal_width[idx], self.testing_petal_length[idx], self.testing_petal_width[idx]])
            if _pred == self.testing_label[idx]:
                hit += 1
        accuracy = hit / self.testing_label.shape[0]
        return accuracy
```
### Visualize
* We can plot sepal (or petal) width, length with different classes to see the distribution of features of training data (or testing data).
```python
class IrisClassifier:
    .
    .
    .
    def visualize_training_set(self, mode):
        # visulize training data
        # mode = [0: sepal, 1: petal]
        assert mode == 0 or mode == 1, "mode should be 0 or 1"
        plt.figure()
        if mode == 0:
            # for sepal
            sepal_lengths = [[], [], []]
            sepal_widths = [[], [], []]
            for idx in range(self.training_label.shape[0]):
                sepal_lengths[int(self.training_label[idx])].append(self.training_sepal_length[idx])
                sepal_widths[int(self.training_label[idx])].append(self.training_sepal_width[idx])
            plt.scatter(sepal_lengths[0], sepal_widths[0], c='red')
            plt.scatter(sepal_lengths[1], sepal_widths[1], c='green')
            plt.scatter(sepal_lengths[2], sepal_widths[2], c='blue')
            plt.legend(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), loc='upper left')
        elif mode == 1:
            # for petal
            petal_lengths = [[], [], []]
            petal_widths = [[], [], []]
            for idx in range(self.training_label.shape[0]):
                petal_lengths[int(self.training_label[idx])].append(self.training_petal_length[idx])
                petal_widths[int(self.training_label[idx])].append(self.training_petal_width[idx])
            plt.scatter(petal_lengths[0], petal_widths[0], c='red')
            plt.scatter(petal_lengths[1], petal_widths[1], c='green')
            plt.scatter(petal_lengths[2], petal_widths[2], c='blue')
            plt.legend(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), loc='upper left')

        plt.show()
```
## How to run
### Prerequisites
* Package
    * scipy.io
    * numpy
    * matplotlib.pyplot
    * platform
* Python
    * python 3
### Execute
* Run
    * in terminal (use command): ```$ python main.py```
    * in IDE (PyCharm, Spyder, etc.): just click the RUN botton
    * Note that the data and ```main.py``` should be in same directory, otherwise you need to change the paths in ```main.py```
    ```python
    def main():
        assert platform.python_version().split('.')[0] == '3', 'python version should be 3.X'
        classifier = IrisClassifier(path_to_training_set='./training.mat', path_to_testing_set='./testing.mat', use_petal_only=True)
        .
        .
        .
    ```
* Visualize
    * Uncomment line 259 or 260 in ```main.py```
    * Note that mode = 0 for visualizing distribution of sepal size, mode = 1 for petal size.
    ```python
    def main():
        assert platform.python_version().split('.')[0] == '3', 'python version should be 3.X'
        classifier = IrisClassifier(path_to_training_set='./training.mat', path_to_testing_set='./testing.mat', use_petal_only=True)
        acc = classifier.calculate_accuracy()
        print("Accuracy on testing set: ", acc)
        # classifier.visualize_training_set(mode=1)
        # classifier.visualize_testing_set(mode=1)
    ```
## Result
### Prediction
![prediction_result](./results/prediction_result.png)
### Visualization
* Sepal size of training data
![training_sepal_distribution](./results/training_sepal_distribution.png)
* Petal size of training data
![training_petal_distribution](./results/training_petal_distribution.png)
* Sepal size of testing data
![testing_sepal_distribution](./results/testing_sepal_distribution.png)
* Petal size of testing data
![testing_petal_distribution](./results/testing_petal_distribution.png)
