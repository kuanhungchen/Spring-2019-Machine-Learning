import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import os
from dataset import Dataset


class Two_Layer_Classifier(object):

    def __init__(self, hidden_num=6, lr=0.01, initial_patience=3):
        self.param = {'W1': 2 * np.random.rand(3, hidden_num),
                      'b1': 2 * np.random.rand(hidden_num),
                      'W2': 2 * np.random.rand(hidden_num, 3),
                      'b2': 2 * np.random.rand(3)}
        self.ch = {}
        self.lr = lr
        self.patience = initial_patience

    def forward(self, x):
        bias = 1
        x = np.append(x, bias)

        z1 = np.dot(x, self.param['W1']) + self.param['b1']
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.param['W2']) + self.param['b2']
        output = self.softmax(z2)

        self.ch['x'] = x
        self.ch['Z1'] = z1
        self.ch['A1'] = a1
        self.ch['Z2'] = z2
        self.ch['output'] = output

        return output

    def predict(self, x, y):
        ans = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(y)):
                data = np.array([x[i][j], y[i][j], 1])
                z1 = np.dot(data, self.param['W1']) + self.param['b1']
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.param['W2']) + self.param['b2']
                output = self.softmax(z2)
                ans[i][j] = np.argmax(output)

        return ans

    def loss(self, logits, label):
        y = np.zeros(3)
        y[label] = 1
        cross_entropy = np.sum(y[i] * np.log10(logits[i]) + (1-y[i]) * np.log10(1-logits[i]) for i in range(len(logits)))
        cross_entropy = - cross_entropy / len(logits)
        self._label = y
        return cross_entropy

    def backward(self):
        dLoss_Y = -(self._label / self.ch['output'] - (1.0 - self._label) / (1.0 - self.ch['output']))
        dY_Z2 = np.array([(np.exp(self.ch['Z2'][0]) * (np.exp(self.ch['Z2'][1]) + np.exp(self.ch['Z2'][2]))) / ((np.sum(np.exp(self.ch['Z2']))) ** 2),
                 (np.exp(self.ch['Z2'][1]) * (np.exp(self.ch['Z2'][0]) + np.exp(self.ch['Z2'][2]))) / ((np.sum(np.exp(self.ch['Z2']))) ** 2),
                 (np.exp(self.ch['Z2'][2]) * (np.exp(self.ch['Z2'][0]) + np.exp(self.ch['Z2'][1]))) / ((np.sum(np.exp(self.ch['Z2']))) ** 2)])
        dZ2_A1 = self.param['W2']
        dA1_Z1 = self.sigmoid(self.ch['Z1']) * (1.0 - self.sigmoid(self.ch['Z1']))
        dZ1_W1 = np.asarray(self.ch['x'])
        dZ2_W2 = self.ch['A1']

        dLoss_Z2 = dLoss_Y * dY_Z2  # (3,)
        dLoss_A1 = np.dot(dZ2_A1, dLoss_Z2)
        dLoss_Z1 = dLoss_A1 * dA1_Z1
        dLoss_W1 = np.reshape(dZ1_W1, (-1, 1)) * np.reshape(dLoss_Z1, (1, -1))
        dLoss_W2 = np.reshape(dZ2_W2, (-1, 1)) * np.reshape(dLoss_Z2, (1, -1))
        dLoss_b1 = dLoss_Z1 * 1
        dLoss_b2 = dLoss_Z2 * 1

        self.param['W1'] = self.param['W1'] - self.lr * dLoss_W1
        self.param['b1'] = self.param['b1'] - self.lr * dLoss_b1
        self.param['W2'] = self.param['W2'] - self.lr * dLoss_W2
        self.param['b2'] = self.param['b2'] - self.lr * dLoss_b2

    def adjust_lr(self, p):
        if p == (self.patience - 1):
            self.lr /= 5
            self.patience = p

    def save_weight(self):
        try:
            filename = 'weight_two_layer_' + str(len(glob.glob(os.path.join('weights', 'two_layer', '*.pkl')))) + '.pkl'
            with open(os.path.join('weights', 'two_layer', filename), 'wb') as f:
                pickle.dump(self.param, f, pickle.HIGHEST_PROTOCOL)
        except:
            print('Saving weights failed.')

    def load_weight(self, path):
        try:
            with open(path, 'rb') as f:
                self.param = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('Path to weight does not exist.')

    @staticmethod
    def sigmoid(x):
        return np.asarray(1 / (1 + np.exp(-x)))

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x


class Three_Layer_Classifier(object):

    def __init__(self, hidden_num=6, lr=0.01, initial_patience=3):
        self.param = {'W1': 2 * np.random.rand(3, hidden_num),
                      'b1': 2 * np.random.rand(hidden_num),
                      'W2': 2 * np.random.rand(hidden_num, hidden_num),
                      'b2': 2 * np.random.rand(hidden_num),
                      'W3': 2 * np.random.rand(hidden_num, 3),
                      'b3': 2 * np.random.rand(3)}
        self.ch = {}
        self.lr = lr
        self.patience = initial_patience

    def forward(self, x):
        bias = 1
        x = np.append(x, bias)

        z1 = np.dot(x, self.param['W1']) + self.param['b1']
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.param['W2']) + self.param['b2']
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.param['W3']) + self.param['b3']
        output = self.softmax(z3)

        self.ch['x'] = x
        self.ch['Z1'] = z1
        self.ch['A1'] = a1
        self.ch['Z2'] = z2
        self.ch['A2'] = a2
        self.ch['Z3'] = z3
        self.ch['output'] = output

        return output

    def predict(self, x, y):
        ans = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(y)):
                data = np.array([x[i][j], y[i][j], 1])
                z1 = np.dot(data, self.param['W1']) + self.param['b1']
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.param['W2']) + self.param['b2']
                output = self.softmax(z2)
                ans[i][j] = np.argmax(output)

        return ans

    def loss(self, logits, label):
        y = np.zeros(3)
        y[label] = 1
        cross_entropy = np.sum(y[i] * np.log10(logits[i]) + (1-y[i]) * np.log10(1-logits[i]) for i in range(len(logits)))
        cross_entropy = - cross_entropy / len(logits)
        self._label = y
        return cross_entropy

    def backward(self):
        dLoss_Y = -(self._label / self.ch['output'] - (1.0 - self._label) / (1.0 - self.ch['output']))
        dY_Z3 = np.array([(np.exp(self.ch['Z3'][0]) * (np.exp(self.ch['Z3'][1]) + np.exp(self.ch['Z3'][2]))) / ((np.sum(np.exp(self.ch['Z3']))) ** 2),
                 (np.exp(self.ch['Z3'][1]) * (np.exp(self.ch['Z3'][0]) + np.exp(self.ch['Z3'][2]))) / ((np.sum(np.exp(self.ch['Z3']))) ** 2),
                 (np.exp(self.ch['Z3'][2]) * (np.exp(self.ch['Z3'][0]) + np.exp(self.ch['Z3'][1]))) / ((np.sum(np.exp(self.ch['Z3']))) ** 2)])
        dZ3_A2 = self.param['W3']
        dA2_Z2 = self.sigmoid(self.ch['Z2']) * (1.0 - self.sigmoid(self.ch['Z2']))
        dZ2_A1 = self.param['W2']
        dA1_Z1 = self.sigmoid(self.ch['Z1']) * (1.0 - self.sigmoid(self.ch['Z1']))
        dZ1_W1 = np.asarray(self.ch['x'])
        dZ2_W2 = self.ch['A1']
        dZ3_W3 = self.ch['A2']

        dLoss_Z3 = dLoss_Y * dY_Z3
        dLoss_A2 = np.dot(dZ3_A2, dLoss_Z3)
        dLoss_Z2 = dLoss_A2 * dA2_Z2
        dLoss_A1 = np.dot(dZ2_A1, dLoss_Z2)
        dLoss_Z1 = dLoss_A1 * dA1_Z1
        dLoss_W1 = np.reshape(dZ1_W1, (-1, 1)) * np.reshape(dLoss_Z1, (1, -1))
        dLoss_W2 = np.reshape(dZ2_W2, (-1, 1)) * np.reshape(dLoss_Z2, (1, -1))
        dLoss_W3 = np.reshape(dZ3_W3, (-1, 1)) * np.reshape(dLoss_Z3, (1, -1))
        dLoss_b1 = dLoss_Z1 * 1
        dLoss_b2 = dLoss_Z2 * 1
        dLoss_b3 = dLoss_Z3 * 1

        self.param['W1'] = self.param['W1'] - self.lr * dLoss_W1
        self.param['b1'] = self.param['b1'] - self.lr * dLoss_b1
        self.param['W2'] = self.param['W2'] - self.lr * dLoss_W2
        self.param['b2'] = self.param['b2'] - self.lr * dLoss_b2
        self.param['W3'] = self.param['W3'] - self.lr * dLoss_W3
        self.param['b3'] = self.param['b3'] - self.lr * dLoss_b3

    def adjust_lr(self, p):
        if p == (self.patience - 1):
            self.lr /= 5
            self.patience = p

    def save_weight(self):
        try:
            filename = 'weight_three_layer_' + str(len(glob.glob(os.path.join('weights', 'three_layer', '*.pkl')))) + '.pkl'
            with open(os.path.join('weights', 'three_layer', filename), 'wb') as f:
                pickle.dump(self.param, f, pickle.HIGHEST_PROTOCOL)
        except:
            print('Saving weights failed.')

    def load_weight(self, path):
        try:
            with open(path, 'rb') as f:
                self.param = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('Path to weight does not exist.')

    @staticmethod
    def sigmoid(x):
        return np.asarray(1 / (1 + np.exp(-x)))

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x


class Decision_Region_Plotter(object):

    def __init__(self, model, dataset):
        self.classifier = model
        self.dataset = dataset
        self.x_range, self.y_range = self.calculate_static()

    def plot(self):
        x = np.linspace(self.x_range.start, self.x_range.stop, 500)
        y = np.linspace(self.y_range.start, self.y_range.stop, 500)
        X, Y = np.meshgrid(x, y)

        plt.figure()
        plt.contourf(x, y, self.classifier.predict(X, Y), 3, alpha=0.75, cmap=plt.cm.hot)
        x_pt = [[], [], []]
        y_pt = [[], [], []]
        for i in range(len(self.dataset)):
            x_pt[self.dataset[i]['label']].append(self.dataset[i]['principal_component'][0])
            y_pt[self.dataset[i]['label']].append(self.dataset[i]['principal_component'][1])
        plt.scatter(x_pt[0], y_pt[0], c='red')
        plt.scatter(x_pt[1], y_pt[1], c='green')
        plt.scatter(x_pt[2], y_pt[2], c='blue')

        plt.show()

    def calculate_static(self):
        x_min = 100
        x_max = -100
        y_min = 100
        y_max = -100
        for i in range(len(self.dataset)):
            pc = self.dataset[i]['principal_component']
            x_min = pc[0] if pc[0] < x_min else x_min
            x_max = pc[0] if pc[0] > x_max else x_max
            y_min = pc[1] if pc[1] < y_min else y_min
            y_max = pc[1] if pc[1] > y_max else y_max

        return range(int(x_min)-1, int(x_max)+1), range(int(y_min)-1, int(y_max)+1)


class Dataset_Visualizer(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def visualize(self):
        colors = ['red', 'green', 'blue']
        plt.figure()
        for i in range(len(self.dataset)):
            pc = self.dataset[i]['principal_component']
            label = self.dataset[i]['label']
            plt.scatter(pc[0], pc[1], c=colors[label])
            plt.legend(('Pear', 'Carambula', 'Lychee'), loc='upper left')
        plt.show()


if __name__ == '__main__':
    def main():
        _dataset = Dataset(os.path.join('data'), mode=Dataset.Mode.TRAIN)
        _dataset_visualizer = Dataset_Visualizer(_dataset)
        _dataset_visualizer.visualize()
    main()
