import csv
import os


with open(os.path.join('fgd_data', 'test5.csv'), 'w', newline='') as test5:
    with open(os.path.join('fgd_data', 'test4.csv'), newline='') as test4:
        with open(os.path.join('fgd_data', 'train.csv'), newline='') as train:
            writer = csv.writer(test5, delimiter=',')
            writer.writerow(['ID','TS','X01','X02','X03','X04','X05','X06','X07','X08','X09','X10','X11','X12','X13',
                             'X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','Y'])

            targets = csv.DictReader(test4)
            sources = csv.DictReader(train)
            target2 = -1
            for target in targets:
                target = int(target['ID']) - 1
                if target2 == target:
                    writer.writerow(
                        [ID, TS, X01, X02, X03, X04, X05, X06, X07, X08, X09, X10, X11, X12, X13, X14, X15, X16, X17,
                         X18, X19, X20, X21, X22, X23, X24, Y])
                    continue
                for source in sources:
                    if int(source['ID']) == target:
                        ID = int(source['ID'])
                        TS = source['TS']
                        X01 = float(source['X01'])
                        X02 = float(source['X02'])
                        X03 = float(source['X03'])
                        X04 = float(source['X04'])
                        X05 = float(source['X05'])
                        X06 = float(source['X06'])
                        X07 = float(source['X07'])
                        X08 = float(source['X08'])
                        X09 = float(source['X09'])
                        X10 = float(source['X10'])
                        X11 = float(source['X11'])
                        X12 = float(source['X12'])
                        X13 = float(source['X13'])
                        X14 = float(source['X14'])
                        X15 = float(source['X15'])
                        X16 = float(source['X16'])
                        X17 = float(source['X17'])
                        X18 = float(source['X18'])
                        X19 = float(source['X19'])
                        X20 = float(source['X20'])
                        X21 = float(source['X21'])
                        X22 = float(source['X22'])
                        X23 = float(source['X23'])
                        X24 = float(source['X24'])
                        Y = float(source['Y'])
                        writer.writerow([ID,TS,X01,X02,X03,X04,X05,X06,X07,X08,X09,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,Y])
                        break
                target2 = target
