import numpy as np
import platform


class Regressor(object):
    def __init__(self, path_to_training_data, path_to_testing_data):
        """
        :param path_to_training_data: path to training set
        :param path_to_testing_data: path to testing set
        """
        self._path_to_training_data = path_to_training_data
        self._path_to_testing_data = path_to_testing_data
        self._training_data, self._testing_data = self.load_data()
        self._static = None
        self._weight = None

    def load_data(self):
        """
        :return: all training data and testing data
        """
        try:
            training_data = np.loadtxt(open(self._path_to_training_data, 'r'), delimiter=',')
        except FileNotFoundError:
            raise FileNotFoundError("cannot find the training data")
        try:
            testing_data = np.loadtxt(open(self._path_to_testing_data, 'r'), delimiter=',')
        except FileNotFoundError:
            raise FileNotFoundError("cannot find the testing data")
        return training_data, testing_data

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


def main():
    assert platform.python_version().split('.')[0] == '3', 'python version should be 3.X'
    _regressor = Regressor(path_to_training_data='./data/Training_set.csv', path_to_testing_data='./data/Testing_set.csv')

    _o1, _o2 = [2, 2]
    _regressor.calculate_static(_o1, _o2)
    _o1, _o2 = [2, 2]
    _regressor.calculate_weight(_o1, _o2)
    print('for (o1, o2) = ({:d}, {:d}), squared error: {:.4f}'.format(_o1, _o2, np.sum(_regressor.calculate_squared_error(_o1, _o2))))

    _example = [321, 112, 1, 0.77]
    print('Feature = [{:d}, {:d}, {:d}], ground truth = [{:.2f}]'.format(int(_example[0]), int(_example[1]), int(_example[2]), _example[3]))
    print('Prediction = [{:.4f}]'.format(_regressor.predict(_example, _o1, _o2)))

    _regressor.visualize(_o1, _o2)


if __name__ == '__main__':
    main()
