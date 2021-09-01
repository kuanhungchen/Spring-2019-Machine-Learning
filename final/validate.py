import csv
import os
import numpy as np
from classifier import Classifier


model1 = Classifier(base=2, random_state=2, n_estimators=300, special=1)
model2 = Classifier(base=2, random_state=2, n_estimators=300, special=2)
model3 = Classifier(base=2, random_state=2, n_estimators=300, special=3)
model4 = Classifier(base=2, random_state=2, n_estimators=300, special=4)
model5 = Classifier(base=2, random_state=2, n_estimators=300, special=5)
with open(os.path.join('fgd_data', 'train.csv'), newline='') as file:
    rows = csv.DictReader(file)
    test_data = []
    hit = 0
    total = 0
    # losses = 0

    for row in rows:
        if int(row['ID']) < 18621:
            continue
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

        gt = int(row['Y'])
        test_data = np.reshape(test_data, (-1, len(test_data)))
        pred1 = model1.predict(test_data)
        pred2 = model2.predict(test_data)
        pred3 = model3.predict(test_data)
        pred4 = model4.predict(test_data)
        pred5 = model5.predict(test_data)
        print(pred1, pred2, pred3, pred4, pred5)
        pred_prob1 = model1.predict_prob(test_data)
        pred_prob2 = model2.predict_prob(test_data)
        pred_prob3 = model3.predict_prob(test_data)
        pred_prob4 = model4.predict_prob(test_data)
        pred_prob5 = model5.predict_prob(test_data)
        print(pred_prob1)
        print(pred_prob2)
        print(pred_prob3)
        print(pred_prob4)
        print(pred_prob5)
        # pred_prob1 = model1.predict_prob(test_data)[1]
        # pred_prob2 = model2.predict_prob(test_data)[1]
        # pred_prob3 = model3.predict_prob(test_data)[1]
        # pred_prob4 = model4.predict_prob(test_data)[1]
        # pred_prob5 = model5.predict_prob(test_data)[1]

        pred_prob = [pred_prob1[1], pred_prob2[1], pred_prob3[1], pred_prob4[1], pred_prob5[1]]
        # print(pred_prob)
        pred = pred_prob.index(np.max(pred_prob)) + 1
        print(pred, " ", gt)

        total += 1
        if int(pred) == gt:
            hit += 1
        # if int(pred) != gt and pred_prob[gt-1] == 0:
        #     loss = -10
        # else:
        #     loss = -np.log10(pred_prob[int(gt)-1])
        # losses += loss
        test_data = []

    print("Hit =  ", hit)
    print("Total = ", total)
    print("Accuracy = ", float(hit / total))
    # print("Loss = ", losses)
    # print("Avg Loss = ", float(losses / total))