import csv
import os
import numpy as np
from classifier import Classifier


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def write_submission(m1, m2, m3, m4, m5):
    filename = os.path.join('result', 'final_b_'+str(m1.base)+'_n_'+str(m1.n_estimators)+'.csv')
    os.mknod(filename)
    with open(os.path.join('fgd_data', 'test.csv'), newline='') as file:
        with open(os.path.join('fgd_data', 'test2.csv'), newline='') as p_file:
            with open(os.path.join('fgd_data', 'test3.csv'), newline='') as pp_file:
                with open(os.path.join('fgd_data', 'test4.csv'), newline='') as ppp_file:
                    with open(os.path.join('fgd_data', 'test5.csv'), newline='') as pppp_file:
                        with open(filename, 'w', newline='') as submission:
                            writer = csv.writer(submission, delimiter=',')
                            writer.writerow(['ID', 'C1', 'C2', 'C3', 'C4', 'C5'])

                            rows = csv.DictReader(file)
                            p_rows = csv.DictReader(p_file)
                            pp_rows = csv.DictReader(pp_file)
                            ppp_rows = csv.DictReader(ppp_file)
                            pppp_rows = csv.DictReader(pppp_file)

                            print("Start predicting")
                            test_data = []
                            for pppp_row in pppp_rows:
                                for ppp_row in ppp_rows:
                                    for pp_row in pp_rows:
                                        for p_row in p_rows:
                                            for row in rows:
                                                test_data.append(float(row['X07']))
                                                test_data.append(float(row['X08']))
                                                test_data.append(float(row['X09']))
                                                test_data.append(float(row['X14']))
                                                test_data.append(float(row['X16']))
                                                test_data.append(float(row['X17']))
                                                test_data.append(float(row['X18']))
                                                test_data.append(float(row['X19']))
                                                test_data.append(float(row['X20']))
                                                test_data.append(float(row['X21']))
                                                test_data.append(float(row['X22']))
                                                test_data.append(float(row['X24']))
                                                break
                                            test_data.append(float(p_row['X07']))
                                            test_data.append(float(p_row['X08']))
                                            test_data.append(float(p_row['X09']))
                                            test_data.append(float(p_row['X14']))
                                            test_data.append(float(p_row['X16']))
                                            test_data.append(float(p_row['X17']))
                                            test_data.append(float(p_row['X18']))
                                            test_data.append(float(p_row['X19']))
                                            test_data.append(float(p_row['X20']))
                                            test_data.append(float(p_row['X21']))
                                            test_data.append(float(p_row['X22']))
                                            test_data.append(float(p_row['X24']))
                                            test_data.append(float(p_row['Y']))
                                            break
                                        test_data.append(float(pp_row['X07']))
                                        test_data.append(float(pp_row['X08']))
                                        test_data.append(float(pp_row['X09']))
                                        test_data.append(float(pp_row['X14']))
                                        test_data.append(float(pp_row['X16']))
                                        test_data.append(float(pp_row['X17']))
                                        test_data.append(float(pp_row['X18']))
                                        test_data.append(float(pp_row['X19']))
                                        test_data.append(float(pp_row['X20']))
                                        test_data.append(float(pp_row['X21']))
                                        test_data.append(float(pp_row['X22']))
                                        test_data.append(float(pp_row['X24']))
                                        test_data.append(float(pp_row['Y']))
                                        break

                                    test_data.append(float(ppp_row['X07']))
                                    test_data.append(float(ppp_row['X08']))
                                    test_data.append(float(ppp_row['X09']))
                                    test_data.append(float(ppp_row['X14']))
                                    test_data.append(float(ppp_row['X16']))
                                    test_data.append(float(ppp_row['X17']))
                                    test_data.append(float(ppp_row['X18']))
                                    test_data.append(float(ppp_row['X19']))
                                    test_data.append(float(ppp_row['X20']))
                                    test_data.append(float(ppp_row['X21']))
                                    test_data.append(float(ppp_row['X22']))
                                    test_data.append(float(ppp_row['X24']))
                                    test_data.append(float(ppp_row['Y']))
                                    break

                                test_data.append(float(pppp_row['X07']))
                                test_data.append(float(pppp_row['X08']))
                                test_data.append(float(pppp_row['X09']))
                                test_data.append(float(pppp_row['X14']))
                                test_data.append(float(pppp_row['X16']))
                                test_data.append(float(pppp_row['X17']))
                                test_data.append(float(pppp_row['X18']))
                                test_data.append(float(pppp_row['X19']))
                                test_data.append(float(pppp_row['X20']))
                                test_data.append(float(pppp_row['X21']))
                                test_data.append(float(pppp_row['X22']))
                                test_data.append(float(pppp_row['X24']))
                                test_data.append(float(pppp_row['Y']))

                                test_data = test_data[0:len(m1.bases)]
                                test_data = np.reshape(test_data, (-1, len(test_data)))

                                pred_prob1 = m1.predict_prob(test_data)[1]
                                pred_prob2 = m2.predict_prob(test_data)[1]
                                pred_prob3 = m3.predict_prob(test_data)[1]
                                pred_prob4 = m4.predict_prob(test_data)[1]
                                pred_prob5 = m5.predict_prob(test_data)[1]
                                pred_prob = [pred_prob1, pred_prob2, pred_prob3, pred_prob4, pred_prob5]
                                pred_prob = softmax(pred_prob)  # not sure if this makes sense

                                # pred_prob[:] = [0.01 if pred_prob[i] == 0 else pred_prob[i] for i in range(len(pred_prob))]

                                writer.writerow([row['ID'], pred_prob[0], pred_prob[1], pred_prob[2], pred_prob[3], pred_prob[4]])
                                test_data = []
                            print("Predicting finished")


if __name__ == '__main__':
    bs = [1, 2, 3, 4]
    ns = [150, 250, 350, 450, 550]
    for b in bs:
        for n in ns:
            _model1 = Classifier(base=b, random_state=2, n_estimators=n, special=1)
            _model2 = Classifier(base=b, random_state=2, n_estimators=n, special=2)
            _model3 = Classifier(base=b, random_state=2, n_estimators=n, special=3)
            _model4 = Classifier(base=b, random_state=2, n_estimators=n, special=4)
            _model5 = Classifier(base=b, random_state=2, n_estimators=n, special=5)
            write_submission(_model1, _model2, _model3, _model4, _model5)
