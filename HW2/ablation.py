import numpy as np
from regressor import Regressor


def main():
    _regressor = Regressor(path_to_training_data='./data/Training_set.csv', path_to_testing_data='./data/Testing_set.csv')
    for _o1 in range(2, 5):
        for _o2 in range(2, 5):
            _regressor.calculate_static(_o1, _o2)
            _regressor.calculate_weight(_o1, _o2)
            print('for (o1, o2) = ({:d}, {:d}), squared error: {:.4f}'.format(_o1, _o2, np.sum(_regressor.calculate_squared_error(_o1, _o2))))

            # print('for (o1, o2) = ({:d}, {:d})'.format(_o1, _o2))
            # _regressor.visualize(_o1, _o2)


if __name__ == '__main__':
    main()
