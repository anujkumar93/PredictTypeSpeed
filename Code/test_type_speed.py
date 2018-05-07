from update_test_data import testBufferBatch
import matplotlib.pyplot as plt
import time
import numpy as np
import feature_extraction_helper as feh
from scipy.signal import medfilt
from torch.autograd import Variable
from net import Net
from multiprocessing.pool import ThreadPool
import torch
import ground_truth_speed as gts
import datetime as dt

sensor_axes = 3
total_plot_samples = 4
MODEL_PATH = '../Data/output/20180507_110446_50epochs_1e-05reg/final_model.pth'
OPTIMIZER_PATH = '../Data/output/20180507_110446_50epochs_1e-05reg/final_optimizer.pth'

samples_lengths = [[0]*total_plot_samples for _ in range(sensor_axes)]
samples_x = [[] for _ in range(sensor_axes)]
samples_y = [[] for _ in range(sensor_axes)]
curr_sample_num = [0]*sensor_axes

DURATION = dt.timedelta(seconds=4)


def update_fig(lines, axes, sensor_data, sensor_num):
    global curr_sample_num, samples_y, samples_x, samples_lengths
    num_samp, num_dim = sensor_data.shape
    if curr_sample_num[sensor_num] == 0:
        samples_y[sensor_num] = np.array(sensor_data)
        samples_x[sensor_num] = np.arange(num_samp)
    elif curr_sample_num[sensor_num] < total_plot_samples:
        samples_y[sensor_num] = np.vstack((samples_y[sensor_num], np.array(sensor_data)))
        samples_x[sensor_num] = np.hstack((samples_x[sensor_num], np.arange(num_samp) + samples_x[sensor_num][-1]))
    else:
        samples_y[sensor_num] = samples_y[sensor_num][samples_lengths[sensor_num][0]:, :]
        samples_x[sensor_num] = samples_x[sensor_num][samples_lengths[sensor_num][0]:]
        samples_lengths[sensor_num][0:-1] = samples_lengths[sensor_num][1:]
        samples_lengths[sensor_num][-1] = 0
        curr_sample_num[sensor_num] -= 1
        samples_y[sensor_num] = np.vstack((samples_y[sensor_num], np.array(sensor_data)))
        samples_x[sensor_num] = np.hstack((samples_x[sensor_num], np.arange(num_samp) + samples_x[sensor_num][-1]))
    samples_lengths[sensor_num][curr_sample_num[sensor_num]] = num_samp
    curr_sample_num[sensor_num] += 1

    for i in range(1, num_dim):
        lines[i-1].set_ydata(samples_y[sensor_num][:, i])
        lines[i-1].set_xdata(samples_x[sensor_num])
    max_y_value = np.max(samples_y[sensor_num][:, 1:])
    axes.set_xlim(samples_x[sensor_num][0], samples_x[sensor_num][-1])
    axes.set_ylim(-2*max_y_value, 2*max_y_value)


def evaluate_live_data(model_live, accel_live, gyro_live, gravity_live):
    lh_accel_input = medfilt(accel_live, [7, 1])
    lh_gyro_input = medfilt(gyro_live, [7, 1])
    lh_gravity_input = medfilt(gravity_live, [7, 1])
    # rh_accel_input = medfilt(rh_accel_input, [7, 1])
    # rh_gyro_input = medfilt(rh_gyro_input, [7, 1])
    # rh_gravity_input = medfilt(rh_gravity_input, [7, 1])
    curr_features = feh.get_analytical_features(lh_accel_input)
    curr_features = np.vstack((curr_features, feh.get_analytical_features(lh_gyro_input)))
    curr_features = np.vstack((curr_features, feh.get_analytical_features(lh_gravity_input)))
    # curr_features = np.vstack((curr_features, feh.get_analytical_features(rh_accel_input)))
    # curr_features = np.vstack((curr_features, feh.get_analytical_features(rh_gyro_input)))
    # curr_features = np.vstack((curr_features, feh.get_analytical_features(rh_gravity_input)))
    curr_features = np.vstack(
        (curr_features, feh.get_analytical_features(feh.get_fft_features(lh_accel_input))))
    curr_features = np.vstack((curr_features, feh.get_analytical_features(feh.get_fft_features(lh_gyro_input))))
    curr_features = np.vstack(
        (curr_features, feh.get_analytical_features(feh.get_fft_features(lh_gravity_input))))
    # curr_features = np.vstack(
    #     (curr_features, feh.get_analytical_features(feh.get_fft_features(rh_accel_input))))
    # curr_features = np.vstack((curr_features, feh.get_analytical_features(feh.get_fft_features(rh_gyro_input))))
    # curr_features = np.vstack(
    #     (curr_features, feh.get_analytical_features(feh.get_fft_features(rh_gravity_input))))
    curr_features = curr_features.reshape((1, -1)).astype(np.float32)
    x = Variable(torch.FloatTensor(curr_features))
    model.eval()
    model.zero_grad()
    return model_live(x)


def get_true_typing_speed(start_time):
    pool = ThreadPool(processes=1)
    speed = pool.apply_async(gts.find_gt_type_speed, (start_time, start_time + DURATION))
    return speed.get(timeout=400)


plt.ion()

fig = plt.figure()
ax_accel = fig.add_subplot(311)
lines_accel = [ax_accel.plot([], [])[0] for _ in range(sensor_axes)]

ax_gyro = fig.add_subplot(312)
lines_gyro = [ax_gyro.plot([], [])[0] for _ in range(sensor_axes)]

ax_gravity = fig.add_subplot(313)
lines_gravity = [ax_gravity.plot([], [])[0] for _ in range(sensor_axes)]

buffer_obj = testBufferBatch()

model = Net(102*sensor_axes)
model.load_state_dict(torch.load(MODEL_PATH))
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))

typing_speed = 0
fig2 = plt.figure()
fig2.suptitle(typing_speed, fontsize=14, fontweight='bold')

while True:
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    accel, gyro, gravity, start_time = buffer_obj.get_new_batch()
    if 0 in accel.shape or 0 in gyro.shape or 0 in gravity.shape:
        continue
    typing_speed = evaluate_live_data(model, accel[:, 1:], gyro[:, 1:], gravity[:, 1:])
    gt_typing_speed = get_true_typing_speed(start_time)
    fig2.suptitle('Typing speed (char/min):\n'+str(max(0,typing_speed[0].data[0])) +
                  '\n True Typing speed (char/min):\n'+str(gt_typing_speed), fontsize=14, fontweight='bold')
    time.sleep(4)
    update_fig(lines_accel, ax_accel, accel, 0)
    update_fig(lines_gyro, ax_gyro, gyro, 1)
    update_fig(lines_gravity, ax_gravity, gravity, 2)

