import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from net import Net
import numpy as np
from data_loader import DataLoader
import sys

EPOCHS = 20
BATCH_SIZE = 50
LR = 1e-5
REG = 1e-5
FILE_PATH = "../Data/pickle/4s_continuous.p"
SAVE_DIR = "../Data/output"
CHECKPOINT_INTERVAL = 5


def train(save_dir=SAVE_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG, file_path=FILE_PATH,
          checkpoint_interval=CHECKPOINT_INTERVAL):
    dlo = DataLoader(batch_size, file_path)

    training_set_size, n_features = dlo.train_data.shape
    model = Net(n_features)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=reg)
    loss_func = nn.MSELoss()

    iters_per_epoch = int(np.ceil(training_set_size / batch_size))
    losses = np.zeros([epochs * iters_per_epoch, ])

    for e in range(epochs):
        model.train()  # to make sure components are in 'train' mode; use model.eval() at 'test' time
        dlo.shuffle_training_set()
        for i in range(iters_per_epoch):
            model.zero_grad()
            x, y = dlo.get_next_training_batch()
            x = Variable(torch.FloatTensor(x))
            y = Variable(torch.FloatTensor(y))
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            losses[i + e * iters_per_epoch] = loss.data[0]
            print('Epoch: {:03}    Iter: {:04}    Loss: {}'.format(e, i, loss.data[0]))
            sys.stdout.flush()
            del x, y, output, loss

        # after every checkpoint_interval epochs: save checkpoint model, save loss curve, display test error
        if (e + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), save_dir+'/model_after_epoch_' + str(e) + '.pth')
            torch.save(optimizer.state_dict(), save_dir+'/optimizer_after_epoch_' + str(e) + '.pth')

            # save loss curve so far
            plt.plot(np.arange(losses.shape[0]) + 1, losses)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.tight_layout()
            plt.savefig(save_dir + '/loss_curve_after_epoch_' + str(e) + '.png')
            plt.close()

            # display test error
            _, test_acc = test(model, dlo)
            print('Test accuracy: {0:.3f}%'.format(test_acc * 100))

    return model, optimizer, losses, dlo


def test(model, dlo):
    model.eval()  # to make sure components are in 'test' mode; use model.train() at 'train' time
    test_set_size = dlo.get_test_set_size()
    dlo.reset_test_batch_counter()
    accuracy = 0
    predictions = []
    for i in range(int(np.ceil(test_set_size / float(dlo.get_batch_size())))):
        model.zero_grad()
        x, y = dlo.get_next_test_batch()
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.FloatTensor(y))
        output = model(x)
        predictions.append(output)
        accuracy += len(y) * batch_accuracy(output, y)

    return torch.cat(predictions).data.numpy(), accuracy/test_set_size


def batch_accuracy(output, y):
    num_samples = len(y)
    correct = 0
    for i in range(num_samples):
        if 0.85*y[i].data.double()[0] <= abs(output[i].data.double()[0]) <= 1.15*y[i].data.double()[0]:
            correct += 1
    return correct/float(num_samples)
