import torch
import numpy as np
import trainer
import datetime
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from net import Net


# OPTIONS
DEBUG = False
CONTINUE_TRAINING = False
MODEL_PATH = 'output/25epoch_gce_model.pth'  # .pth file for existing model if continuing training
OPTIMIZER_PATH = 'optimizer.pth'  # .pth file for existing optimizer if continuing training
EPOCHS = 50
BATCH_SIZE = 40
LR = 1e-5
REG = 1e-5
FILE_PATH = "../Data/pickle/4s_left_continuous.p"
SAVE_DIR = "../Data/output"
CHECKPOINT_INTERVAL = 10  # number of epochs between checkpoints (save model and loss curve)
NUM_TEST_SAMPLES = 20  # for generating test samples at the end


# GENERATE SAVE DIRECTORY PATH
timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
if not os.path.exists('../Data/output'):
    os.makedirs('../Data/output')
save_dir = '../Data/output/' + timestamp + '_' + str(EPOCHS) + 'epochs_' + str(REG) + 'reg'
if DEBUG:
    save_dir += '_debug'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# TRAIN AND SAVE MODEL AND OPTIMIZER
if CONTINUE_TRAINING:
    model = FCN()
    model.load_state_dict(torch.load(MODEL_PATH))

    if os.path.exists(OPTIMIZER_PATH):
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    else:
        optimizer = None

    model, optimizer, losses, dlo = trainer.train(save_dir, model=model, optimizer=optimizer,
                                       epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG,
                                       checkpoint_interval=CHECKPOINT_INTERVAL, debug=DEBUG)
else:
    model, optimizer, losses, dlo = trainer.train(save_dir=save_dir, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, lr=LR, reg=REG, file_path=FILE_PATH,
                    checkpoint_interval=CHECKPOINT_INTERVAL)
torch.save(model.state_dict(), save_dir+'/final_model.pth')  # only saves parameters
torch.save(optimizer.state_dict(), save_dir+'/final_optimizer.pth')
# load them back using:
# model = FCN()
# model.load_state_dict(torch.load('model.pth'))
# optimizer = torch.optim.Adam()
# optimizer.load_state_dict(torch.load('optimizer.pth'))


# LOSS CURVE
plt.plot(np.arange(losses.shape[0]) + 1, losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(save_dir+'/final_loss_curve.png')
plt.close()


# TEST THE MODEL
test_preds, test_acc = trainer.test(model, dlo)
print('Final test accuracy: {0:.3f}%'.format(test_acc * 100))

test_set_size=dlo.get_test_set_size()
plt.plot(range(test_set_size), test_preds, 'r-', range(test_set_size), dlo.test_labels, 'b-')
plt.xlabel('Samples')
plt.ylabel('Char/min')
plt.tight_layout()
plt.savefig(save_dir+'/pred_vs_gt.png')
plt.close()