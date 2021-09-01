import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Two_Layer_Classifier, Three_Layer_Classifier
from collections import deque


def _train(path_to_data_dir):

    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    dataset_val = Dataset(path_to_data_dir, mode=Dataset.Mode.VAL)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    dataset_test = Dataset(path_to_data_dir, mode=Dataset.Mode.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    model = Two_Layer_Classifier(hidden_num=6, lr=0.01)
    step = 0
    best_val_loss = 100
    patience = 1
    losses = deque(maxlen=1000)
    should_stop = False

    while not should_stop:
        for batch_idx, data in enumerate(dataloader):
            pc = data['principal_component']
            label = data['label']
            logits = model.forward(pc)
            loss = model.loss(logits, label)
            model.backward()
            losses.append(loss)
            step += 1

            if step % 1000 == 0:
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Avg. Loss = {(avg_loss):.4f}')

            if step % 10000 == 0:
                loss_v = []
                for _, d in enumerate(dataloader_val):
                    pc = d['principal_component']
                    label = d['label']
                    logits = model.forward(pc)

                    loss = model.loss(logits, label)
                    loss_v.append(loss)

                avg_loss = sum(loss_v) / len(loss_v)
                print(f'Val Loss = {(avg_loss):.4f}')
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                else:
                    patience -= 1
                if patience == 0:
                    should_stop = True
                break

    # train
    hit_train = 0
    total_train = 0
    for _, d in enumerate(dataloader):
        pc = d['principal_component']
        label = d['label']
        logits = model.forward(pc)

        pred = np.argmax(logits)
        if pred == label.item():
            hit_train += 1
        total_train += 1
    print(f'Training Acc. = {float(hit_train/total_train):.6f}')

    # val
    hit_val = 0
    total_val = 0
    for _, d in enumerate(dataloader_val):
        pc = d['principal_component']
        label = d['label']
        logits = model.forward(pc)

        pred = np.argmax(logits)
        if pred == label.item():
            hit_val += 1
        total_val += 1
    print(f'Validation Acc. = {float(hit_val/total_val):.6f}')

    # test
    hit_test = 0
    total_test = 0
    for _, d in enumerate(dataloader_test):
        pc = d['principal_component']
        label = d['label']
        logits = model.forward(pc)

        pred = np.argmax(logits)
        if pred == label.item():
            hit_test += 1
        total_test += 1
    print(f'Testing Acc. = {float(hit_test/total_test):.6f}')
    # if float(hit_test/total_test) > 0.2:
    #     model.save_weight()
    model.save_weight()


if __name__ == '__main__':
    def main():
        path_to_data_dir = os.path.join('data')
        _train(path_to_data_dir)
    main()
