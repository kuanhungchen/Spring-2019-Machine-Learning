import scipy.io
import numpy as np
import matplotlib.pyplot as plt


class IrisClassifier:
    def __init__(self, path_to_training_set, path_to_testing_set,
                 use_petal_only=False):
        self.path_to_training_set = path_to_training_set
        self.path_to_testing_set = path_to_testing_set

        self.total_num_of_training_set, self.prior_prob, \
            self.training_sepal_length, self.training_sepal_width, \
            self.training_petal_length, self.training_petal_width, \
            self.training_label = self.prepare_training_data()
        self.testing_sepal_length, self.testing_sepal_width, \
            self.testing_petal_length, self.testing_petal_width, \
            self.testing_label = self.prepare_testing_data()

        self.statics = self.calculate_statics()
        self.use_petal_only = use_petal_only

    def prepare_training_data(self):
        training_data = scipy.io.loadmat(self.path_to_training_set)

        training_sepal_length = np.asarray(
            training_data[list(training_data.keys())[3]][0][:])
        training_sepal_width = np.asarray(
            training_data[list(training_data.keys())[4]][0][:])
        training_petal_length = np.asarray(
            training_data[list(training_data.keys())[5]][0][:])
        training_petal_width = np.asarray(
            training_data[list(training_data.keys())[6]][0][:])
        training_label = np.asarray(
            training_data[list(training_data.keys())[7]][0][:])

        # calculate prior probability for each class
        prior_prob = [0, 0, 0]
        for label in training_label:
            prior_prob[int(label)] += 1
        total_num_of_training_set = sum(prior_prob)
        prior_prob = [p / total_num_of_training_set for p in prior_prob]

        return total_num_of_training_set, prior_prob, training_sepal_length, \
            training_sepal_width, training_petal_length, \
            training_petal_width, training_label

    def prepare_testing_data(self):
        testing_data = scipy.io.loadmat(self.path_to_testing_set)

        testing_sepal_length = np.asarray(
            testing_data[list(testing_data.keys())[3]][0][:])
        testing_sepal_width = np.asarray(
            testing_data[list(testing_data.keys())[4]][0][:])
        testing_petal_length = np.asarray(
            testing_data[list(testing_data.keys())[5]][0][:])
        testing_petal_width = np.asarray(
            testing_data[list(testing_data.keys())[6]][0][:])
        testing_label = np.asarray(
            testing_data[list(testing_data.keys())[7]][0][:])

        return testing_sepal_length, testing_sepal_width, \
            testing_petal_length, testing_petal_width, testing_label

    def calculate_statics(self):
        # calculate mean and standard deviation for each class
        sepal_length_mean = [0, 0, 0]
        for idx, sepal_length in enumerate(self.training_sepal_length):
            sepal_length_mean[int(self.training_label[idx])] += sepal_length
        sepal_length_mean = [
            m / (self.total_num_of_training_set * self.prior_prob[i])
            for i, m in enumerate(sepal_length_mean)]

        sepal_length_stdev = [0, 0, 0]
        for idx, sepal_length in enumerate(self.training_sepal_length):
            sepal_length_stdev[int(self.training_label[idx])] += (
                sepal_length - sepal_length_mean[int(self.training_label[idx])]
            ) ** 2
        sepal_length_stdev = [
            stdev / (self.total_num_of_training_set * self.prior_prob[i])
            for i, stdev in enumerate(sepal_length_stdev)]
        sepal_length_stdev = [stdev ** 0.5 for stdev in sepal_length_stdev]

        sepal_width_mean = [0, 0, 0]
        for idx, sepal_width in enumerate(self.training_sepal_width):
            sepal_width_mean[int(self.training_label[idx])] += sepal_width
        sepal_width_mean = [
            m / (self.total_num_of_training_set * self.prior_prob[i])
            for i, m in enumerate(sepal_width_mean)]

        sepal_width_stdev = [0, 0, 0]
        for idx, sepal_width in enumerate(self.training_sepal_width):
            sepal_width_stdev[int(self.training_label[idx])] += (
                sepal_width - sepal_width_mean[int(self.training_label[idx])]
            ) ** 2
        sepal_width_stdev = [
            stdev / (self.total_num_of_training_set * self.prior_prob[i])
            for i, stdev in enumerate(sepal_width_stdev)]
        sepal_width_stdev = [stdev ** 0.5 for stdev in sepal_width_stdev]

        petal_length_mean = [0, 0, 0]
        for idx, petal_length in enumerate(self.training_petal_length):
            petal_length_mean[int(self.training_label[idx])] += petal_length
        petal_length_mean = [
            m / (self.total_num_of_training_set * self.prior_prob[i])
            for i, m in enumerate(petal_length_mean)]

        petal_length_stdev = [0, 0, 0]
        for idx, petal_length in enumerate(self.training_petal_length):
            petal_length_stdev[int(self.training_label[idx])] += (
                petal_length - petal_length_mean[int(self.training_label[idx])]
            ) ** 2
        petal_length_stdev = [
            stdev / (self.total_num_of_training_set * self.prior_prob[i])
            for i, stdev in enumerate(petal_length_stdev)]
        petal_length_stdev = [stdev ** 0.5 for stdev in petal_length_stdev]

        petal_width_mean = [0, 0, 0]
        for idx, petal_width in enumerate(self.training_petal_width):
            petal_width_mean[int(self.training_label[idx])] += petal_width
        petal_width_mean = [
            m / (self.total_num_of_training_set * self.prior_prob[i])
            for i, m in enumerate(petal_width_mean)]

        petal_width_stdev = [0, 0, 0]
        for idx, petal_width in enumerate(self.training_petal_width):
            petal_width_stdev[int(self.training_label[idx])] += (
                petal_width - petal_width_mean[int(self.training_label[idx])]
            ) ** 2
        petal_width_stdev = [
            stdev / (self.total_num_of_training_set * self.prior_prob[i])
            for i, stdev in enumerate(petal_width_stdev)]
        petal_width_stdev = [stdev ** 0.5 for stdev in petal_width_stdev]

        statics = [[[], [], []] for _ in range(3)]
        for c in range(3):
            statics[c][0].append(sepal_length_mean[c])
            statics[c][0].append(sepal_length_stdev[c])
            statics[c][1].append(sepal_width_mean[c])
            statics[c][1].append(sepal_width_stdev[c])
            statics[c][2].append(petal_length_mean[c])
            statics[c][2].append(petal_length_stdev[c])
            statics[c][3].append(petal_width_mean[c])
            statics[c][3].append(petal_width_stdev[c])
        return statics

    @staticmethod
    def normal_pdf(x, mean, stdev):
        # compute probability based on a normal PDF
        variance = stdev ** 2
        exp_squared_diff = (x - mean) ** 2
        exp_power = - exp_squared_diff / (2 * variance)
        exponent = np.e ** exp_power
        denominator = ((2 * np.pi) ** 0.5) * stdev
        normal_prob = exponent / denominator
        return normal_prob

    def calculate_joint_prob(self, input_feature):
        # calculate corresponding joint probability for each class
        # given a new data
        joint_probs = [0, 0, 0]
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

    @staticmethod
    def marginal_pdf(joint_probabilities):
        # compute marginal probability
        marginal_prob = sum(joint_probabilities[:])
        return marginal_prob

    def calculate_posterior_probs(self, input_feature):
        # calculate corresponding posterior probability for each class
        # given a new data
        posterior_probs = [0, 0, 0]
        joint_probabilities = self.calculate_joint_prob(input_feature)
        marginal_prob = self.marginal_pdf(joint_probabilities)
        for idx, _joint_prob in enumerate(joint_probabilities):
            posterior_probs[idx] = _joint_prob / marginal_prob
        return posterior_probs

    def predict(self, input_feature):
        # pick maximum posterior prob. (i.e. MAP) as prediction
        posterior_probs = self.calculate_posterior_probs(input_feature)
        map_prob = max(posterior_probs[:])
        map_idx = posterior_probs.index(map_prob)
        return map_prob, map_idx

    def calculate_accuracy(self):
        # calculate prediction accuracy
        hit = 0
        for idx in range(self.testing_label.shape[0]):
            _, _pred = self.predict(
                [
                    self.testing_sepal_length[idx],
                    self.testing_sepal_width[idx],
                    self.testing_petal_length[idx],
                    self.testing_petal_width[idx]
                ])
            if _pred == self.testing_label[idx]:
                hit += 1
        accuracy = hit / self.testing_label.shape[0]
        return accuracy

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
                sepal_lengths[int(self.training_label[idx])].append(
                    self.training_sepal_length[idx])
                sepal_widths[int(self.training_label[idx])].append(
                    self.training_sepal_width[idx])
            plt.scatter(sepal_lengths[0], sepal_widths[0], c='red')
            plt.scatter(sepal_lengths[1], sepal_widths[1], c='green')
            plt.scatter(sepal_lengths[2], sepal_widths[2], c='blue')
            plt.legend(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                       loc='upper left')
        elif mode == 1:
            # for petal
            petal_lengths = [[], [], []]
            petal_widths = [[], [], []]
            for idx in range(self.training_label.shape[0]):
                petal_lengths[int(self.training_label[idx])].append(
                    self.training_petal_length[idx])
                petal_widths[int(self.training_label[idx])].append(
                    self.training_petal_width[idx])
            plt.scatter(petal_lengths[0], petal_widths[0], c='red')
            plt.scatter(petal_lengths[1], petal_widths[1], c='green')
            plt.scatter(petal_lengths[2], petal_widths[2], c='blue')
            plt.legend(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                       loc='upper left')
        plt.show()

    def visualize_testing_set(self, mode):
        # visualize testing data
        # mode = [0: sepal, 1: petal]
        assert mode == 0 or mode == 1, "mode should be 0 or 1"
        plt.figure()
        if mode == 0:
            # for sepal
            sepal_lengths = [[], [], []]
            sepal_widths = [[], [], []]
            for idx in range(self.testing_label.shape[0]):
                sepal_lengths[int(self.testing_label[idx])].append(
                    self.testing_sepal_length[idx])
                sepal_widths[int(self.testing_label[idx])].append(
                    self.testing_sepal_width[idx])
            plt.scatter(sepal_lengths[0], sepal_widths[0], c='red')
            plt.scatter(sepal_lengths[1], sepal_widths[1], c='green')
            plt.scatter(sepal_lengths[2], sepal_widths[2], c='blue')
            plt.legend(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                       loc='upper left')
        elif mode == 1:
            # for petal
            petal_lengths = [[], [], []]
            petal_widths = [[], [], []]
            for idx in range(self.testing_label.shape[0]):
                petal_lengths[int(self.testing_label[idx])].append(
                    self.testing_petal_length[idx])
                petal_widths[int(self.testing_label[idx])].append(
                    self.testing_petal_width[idx])
            plt.scatter(petal_lengths[0], petal_widths[0], c='red')
            plt.scatter(petal_lengths[1], petal_widths[1], c='green')
            plt.scatter(petal_lengths[2], petal_widths[2], c='blue')
            plt.legend(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                       loc='upper left')
        plt.show()


def main():
    classifier = IrisClassifier(path_to_training_set='./training.mat',
                                path_to_testing_set='./testing.mat',
                                use_petal_only=True)
    # compute accuracy on testing set
    acc = classifier.calculate_accuracy()
    print("Accuracy on testing set: ", acc)

    # visualize
    # classifier.visualize_training_set(mode=1)
    # classifier.visualize_testing_set(mode=1)


if __name__ == '__main__':
    main()
