import numpy as np
import csv
import os
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier


class Classifier(object):

    def __init__(self, base=0, criterion='entropy', random_state=5, n_estimators=340, min_samples_split=2,
                 special=None, data_split=20622):
        super(Classifier, self).__init__()
        self.base = base
        self.criterion = criterion
        assert self.criterion in ['entropy', 'gini'], 'criterion should be entropy or gini'
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.special = special
        if self.special is not None:
            assert self.special in [1, 2, 3, 4, 5], 'special should between 1 and 5'
            assert self.random_state == 2, 'random_state should be 2 for multiple classifiers'
        self.data_split = data_split

        llll_range, lll_range, ll_range, l_range, now_range = self._range()

        self.model = RandomForestClassifier(criterion=self.criterion,
                                            random_state=self.random_state,
                                            n_estimators=self.n_estimators,
                                            min_samples_split=self.min_samples_split,
                                            oob_score=True)

        print("Classifier created")

        with open(os.path.join('fgd_data', 'train.csv'), newline='') as file:
            rows = csv.DictReader(file)
            data = defaultdict(list)
            for row in rows:
                id = int(row['ID'])
                if id in llll_range:
                    data['last_last_last_last_X07'].append(float(row['X07']))
                    data['last_last_last_last_X08'].append(float(row['X08']))
                    data['last_last_last_last_X09'].append(float(row['X09']))
                    data['last_last_last_last_X14'].append(float(row['X14']))
                    data['last_last_last_last_X16'].append(float(row['X14']))
                    data['last_last_last_last_X17'].append(float(row['X14']))
                    data['last_last_last_last_X18'].append(float(row['X14']))
                    data['last_last_last_last_X19'].append(float(row['X14']))
                    data['last_last_last_last_X20'].append(float(row['X20']))
                    data['last_last_last_last_X21'].append(float(row['X21']))
                    data['last_last_last_last_X22'].append(float(row['X22']))
                    data['last_last_last_last_X24'].append(float(row['X24']))
                    data['last_last_last_last_gt'].append(float(row['Y']))
                if id in lll_range:
                    data['last_last_last_X07'].append(float(row['X07']))
                    data['last_last_last_X08'].append(float(row['X08']))
                    data['last_last_last_X09'].append(float(row['X09']))
                    data['last_last_last_X14'].append(float(row['X14']))
                    data['last_last_last_X16'].append(float(row['X14']))
                    data['last_last_last_X17'].append(float(row['X14']))
                    data['last_last_last_X18'].append(float(row['X14']))
                    data['last_last_last_X19'].append(float(row['X14']))
                    data['last_last_last_X20'].append(float(row['X20']))
                    data['last_last_last_X21'].append(float(row['X21']))
                    data['last_last_last_X22'].append(float(row['X22']))
                    data['last_last_last_X24'].append(float(row['X24']))
                    data['last_last_last_gt'].append(float(row['Y']))
                if id in ll_range:
                    data['last_last_X07'].append(float(row['X07']))
                    data['last_last_X08'].append(float(row['X08']))
                    data['last_last_X09'].append(float(row['X09']))
                    data['last_last_X14'].append(float(row['X14']))
                    data['last_last_X16'].append(float(row['X14']))
                    data['last_last_X17'].append(float(row['X14']))
                    data['last_last_X18'].append(float(row['X14']))
                    data['last_last_X19'].append(float(row['X14']))
                    data['last_last_X20'].append(float(row['X20']))
                    data['last_last_X21'].append(float(row['X21']))
                    data['last_last_X22'].append(float(row['X22']))
                    data['last_last_X24'].append(float(row['X24']))
                    data['last_last_gt'].append(float(row['Y']))
                if id in l_range:
                    data['last_X07'].append(float(row['X07']))
                    data['last_X08'].append(float(row['X08']))
                    data['last_X09'].append(float(row['X09']))
                    data['last_X14'].append(float(row['X14']))
                    data['last_X16'].append(float(row['X14']))
                    data['last_X17'].append(float(row['X14']))
                    data['last_X18'].append(float(row['X14']))
                    data['last_X19'].append(float(row['X14']))
                    data['last_X20'].append(float(row['X20']))
                    data['last_X21'].append(float(row['X21']))
                    data['last_X22'].append(float(row['X22']))
                    data['last_X24'].append(float(row['X24']))
                    data['last_gt'].append(float(row['Y']))
                if id in now_range:
                    data['X07'].append(float(row['X07']))
                    data['X08'].append(float(row['X08']))
                    data['X09'].append(float(row['X09']))
                    data['X14'].append(float(row['X14']))
                    data['X16'].append(float(row['X16']))
                    data['X17'].append(float(row['X17']))
                    data['X18'].append(float(row['X18']))
                    data['X19'].append(float(row['X19']))
                    data['X20'].append(float(row['X20']))
                    data['X21'].append(float(row['X21']))
                    data['X22'].append(float(row['X22']))
                    data['X24'].append(float(row['X24']))
                    if self.special is not None:
                        data['gt'].append(2 if self.special == int(row['Y']) else 1)
                    else:
                        data['gt'].append(int(row['Y']))

        if self.base == 0:  # 12
            self.bases = ['X07', 'X08', 'X09', 'X14', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X24']
        elif self.base == 1:  # 25
            self.bases = ['X07', 'X08', 'X09', 'X14', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X24',
                          'last_X07', 'last_X08', 'last_X09', 'last_X14', 'last_X16', 'last_X17', 'last_X18',
                          'last_X19', 'last_X20', 'last_X21', 'last_X22', 'last_X24', 'last_gt']
        elif self.base == 2:  # 38
            self.bases = ['X07', 'X08', 'X09', 'X14', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X24',
                          'last_X07', 'last_X08', 'last_X09', 'last_X14', 'last_X16', 'last_X17', 'last_X18',
                          'last_X19', 'last_X20', 'last_X21', 'last_X22', 'last_X24', 'last_gt',
                          'last_last_X07', 'last_last_X08', 'last_last_X09', 'last_last_X14', 'last_last_X16',
                          'last_last_X17', 'last_last_X18', 'last_last_X19', 'last_last_X20', 'last_last_X21',
                          'last_last_X22', 'last_last_X24', 'last_last_gt']
        elif self.base == 3:  # 51
            self.bases = ['X07', 'X08', 'X09', 'X14', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X24',
                          'last_X07', 'last_X08', 'last_X09', 'last_X14', 'last_X16', 'last_X17', 'last_X18',
                          'last_X19', 'last_X20', 'last_X21', 'last_X22', 'last_X24', 'last_gt',
                          'last_last_X07', 'last_last_X08', 'last_last_X09', 'last_last_X14', 'last_last_X16',
                          'last_last_X17', 'last_last_X18', 'last_last_X19', 'last_last_X20', 'last_last_X21',
                          'last_last_X22', 'last_last_X24', 'last_last_gt',
                          'last_last_last_X07', 'last_last_last_X08', 'last_last_last_X09', 'last_last_last_X14',
                          'last_last_last_X16', 'last_last_last_X17', 'last_last_last_X18', 'last_last_last_X19',
                          'last_last_last_X20', 'last_last_last_X21', 'last_last_last_X22', 'last_last_last_X24',
                          'last_last_last_gt']
        elif self.base == 4:  # 64
            self.bases = ['X07', 'X08', 'X09', 'X14', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X24',
                          'last_X07', 'last_X08', 'last_X09', 'last_X14', 'last_X16', 'last_X17', 'last_X18',
                          'last_X19', 'last_X20', 'last_X21', 'last_X22', 'last_X24', 'last_gt',
                          'last_last_X07', 'last_last_X08', 'last_last_X09', 'last_last_X14', 'last_last_X16',
                          'last_last_X17', 'last_last_X18', 'last_last_X19', 'last_last_X20', 'last_last_X21',
                          'last_last_X22', 'last_last_X24', 'last_last_gt',
                          'last_last_last_X07', 'last_last_last_X08', 'last_last_last_X09', 'last_last_last_X14',
                          'last_last_last_X16', 'last_last_last_X17', 'last_last_last_X18', 'last_last_last_X19',
                          'last_last_last_X20', 'last_last_last_X21', 'last_last_last_X22', 'last_last_last_X24',
                          'last_last_last_gt',
                          'last_last_last_last_X07', 'last_last_last_last_X08', 'last_last_last_last_X09', 'last_last_last_last_X14',
                          'last_last_last_last_X16', 'last_last_last_last_X17', 'last_last_last_last_X18', 'last_last_last_last_X19',
                          'last_last_last_last_X20', 'last_last_last_last_X21', 'last_last_last_last_X22', 'last_last_last_last_X24',
                          'last_last_last_last_gt']
        else:
            raise ValueError('base choice must be 0, 1, 2, 3 or 4')

        X = []
        for base in self.bases:
            tmp = np.reshape(data[base], (-1, 1))
            try:
                X = np.concatenate((X, tmp), axis=1)
            except ValueError:
                X = tmp

        y = data['gt']
        print("Feature shape ==> ", X.shape)
        print("Label shape ==> ", len(y))

        print("Start fitting classifier")
        self.model.fit(X, y)
        print("Classifier fitting finished")
        self.oob_score = self.model.oob_score_
        # print(model.oob_score_)

    def _range(self):
        llll_range = np.arange(-2, -1)
        lll_range = np.arange(-2, -1)
        ll_range = np.arange(-2, -1)
        l_range = np.arange(-2, -1)
        if self.base == 0:
            now_range = np.arange(1, self.data_split)
        elif self.base == 1:
            l_range = np.arange(1, self.data_split-1)
            now_range = np.arange(2, self.data_split)
        elif self.base == 2:
            ll_range = np.arange(1, self.data_split-2)
            l_range = np.arange(2, self.data_split-1)
            now_range = np.arange(3, self.data_split)
        elif self.base == 3:
            lll_range = np.arange(1, self.data_split-3)
            ll_range = np.arange(2, self.data_split-2)
            l_range = np.arange(3, self.data_split-1)
            now_range = np.arange(4, self.data_split)
        elif self.base == 4:
            llll_range = np.arange(1, self.data_split-4)
            lll_range = np.arange(2, self.data_split-3)
            ll_range = np.arange(3, self.data_split-2)
            l_range = np.arange(4, self.data_split-1)
            now_range = np.arange(5, self.data_split)
        else:
            raise ValueError('base should be 0, 1, 2, 3 or 4')
        return llll_range, lll_range, ll_range, l_range, now_range

    def predict(self, x):
        return self.model.predict(x)

    def predict_prob(self, x):
        return self.model.predict_proba(x)[0]

    def description(self):
        description = {'criterion': self.criterion, 'n_estimators': self.n_estimators,
                       'min_samples_split': self.min_samples_split, 'oob_score': self.oob_score,
                       'special': self.special, 'data_split': self.data_split, 'bases': self.bases}
        return description


def main():
    _example1 = Classifier(base=2, n_estimators=250, criterion='gini')
    print(_example1.description())


if __name__ == '__main__':
    main()
