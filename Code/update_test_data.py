import csv
import datetime as dt
import time
import numpy as np
from multiprocessing.pool import ThreadPool

BUFFER = dt.timedelta(microseconds=800000)
DURATION = dt.timedelta(seconds=4)
LEFT_WATCH_FOLDER = '../Data/syncMobileLeft/'


class testBufferBatch:
    def __init__(self):
        self.buffer = BUFFER
        self.duration = DURATION
        self.left_accel_file = LEFT_WATCH_FOLDER+'Accel.csv'
        self.left_gyro_file = LEFT_WATCH_FOLDER+'Gyro.csv'
        self.left_gravity_file = LEFT_WATCH_FOLDER+'Gravity.csv'
        self.curr_start_time = dt.datetime(2017, 12, 31)
        self.next_start_time = dt.datetime(2017, 12, 31)
        self.left_accel_lines = []
        self.left_gyro_lines = []
        self.left_gravity_lines = []

    def get_new_batch(self):
        self.curr_start_time = self.next_start_time
        # if self.left_accel_lines and self.left_gyro_lines and self.left_gravity_lines:
        #     ans = np.array(self.left_accel_lines), np.array(self.left_gyro_lines),\
        #           np.array(self.left_gravity_lines), self.curr_start_time
        #     self.left_accel_lines = []
        #     self.left_gyro_lines = []
        #     self.left_gravity_lines = []
        #     self.next_start_time += self.duration
        #     return ans

        self.curr_start_time = self.get_common_start_time((self.left_accel_file, self.left_gyro_file,\
                                                           self.left_gravity_file))

        pool = ThreadPool(processes=3)
        left_accel_lines = pool.apply_async(self.get_chunk_data, (self.left_accel_file,
                                            self.curr_start_time, self.curr_start_time + self.duration, self.buffer))
        left_gyro_lines = pool.apply_async(self.get_chunk_data, (self.left_gyro_file,
                                            self.curr_start_time, self.curr_start_time + self.duration, self.buffer))
        left_gravity_lines = pool.apply_async(self.get_chunk_data, (self.left_gravity_file,
                                            self.curr_start_time, self.curr_start_time + self.duration, self.buffer))

        return np.array(left_accel_lines.get(timeout=8)), np.array(left_gyro_lines.get(timeout=8)), \
            np.array(left_gravity_lines.get(timeout=8)), self.curr_start_time

    def get_chunk_data(self, filename, start_time, end_time, log_buffer):
        result_set = []
        count = 0
        while not result_set and count < 30:
            try:
                with open(filename) as f:
                    reader = csv.reader(f)
                    lines = list(reader)
                    num_lines = 0
                    for i in range(len(lines)):
                        curr_time_str = lines[i][1]+" "+lines[i][2]
                        curr_time = dt.datetime.strptime(curr_time_str, ' %b %d %Y %H:%M:%S.%f')
                        if start_time <= curr_time <= end_time:
                            curr_set = [curr_time]
                            curr_set.extend([float(string) for string in lines[i][3:]])
                            result_set.append(curr_set)
                        # elif curr_time > end_time and num_lines == 0:
                        #     num_lines = i
                        if curr_time > end_time+2*log_buffer: #due to discrepancy in data of lower values coming after higher values
                            break
                    # if num_lines > 0:
                    #     new_lines = []
                    #     for i in range(num_lines, len(lines)):
                    #         curr_time_str = lines[i][1] + " " + lines[i][2]
                    #         curr_time = dt.datetime.strptime(curr_time_str, ' %b %d %Y %H:%M:%S.%f')
                    #         curr_set = [curr_time]
                    #         curr_set.extend([float(string) for string in lines[i][3:]])
                    #         new_lines.append(curr_set)
                    #     if new_lines:
                    #         if new_lines[-1][0] - new_lines[0][0] > dt.timedelta(seconds=1):
                    #             if "accel" in filename.lower():
                    #                 self.left_accel_lines = new_lines
                    #             elif "gyro" in filename.lower():
                    #                 self.left_gyro_lines = new_lines
                    #             elif "grav" in filename.lower():
                    #                 self.left_gravity_lines = new_lines
                    #             self.next_start_time = max(self.next_start_time, new_lines[0][0])
            except:
                print("Unable to open file for reading data,", filename, ". Tried", count, "times. Trying again")
            time.sleep(0.1)
            count += 1
        if count >= 30:
            print("Tried 30 times to get data without success for file : " + str(filename))
        self.next_start_time = end_time
        if result_set and result_set[-1][0] - result_set[0][0] < dt.timedelta(seconds=self.buffer.seconds-1):
            result_set = []
        return result_set

    def get_start_time(self, filename):
        count = 0
        curr_time = self.curr_start_time
        while count < 30:
            try:
                with open(filename) as f:
                    reader = csv.reader(f)
                    line = next(reader)
                    curr_time_str = line[1] + " " + line[2]
                    curr_time = dt.datetime.strptime(curr_time_str, ' %b %d %Y %H:%M:%S.%f')
                    break
            except:
                print("Unable to open file for reading start_time,", filename, ". Tried", count, "times. Trying again")
            time.sleep(0.1)
            count += 1
        if count >= 30:
            print("Tried 30 times to get start_time without success for file : " + str(filename))
        return curr_time

    def get_common_start_time(self, left_sensor_files):
        pool = ThreadPool(processes=3)
        start_times = [None] * len(left_sensor_files)
        for i, filename in enumerate(left_sensor_files):
            start_times[i] = pool.apply_async(self.get_start_time, (filename,))
        ans = max(start_time.get(timeout=5) for start_time in start_times)
        ans = max(ans, self.curr_start_time)
        return ans

# b = testBufferBatch()
# acc, gyr, gra, tym = b.get_new_batch()
# print("acc shape", len(acc))
# print("gra shape", len(gra))
# print("gyr shape", len(gyr))
# print(tym, "\n")
#
# acc,gyr,gra,tym = b.get_new_batch()
# print("acc shape", len(acc))
# print("gra shape", len(gra))
# print("gyr shape", len(gyr))
# print(tym,"\n")
#
# acc,gyr,gra,tym = b.get_new_batch()
# print("acc shape", len(acc))
# print("gra shape", len(gra))
# print("gyr shape", len(gyr))
# print(tym,"\n")
#
# acc,gyr,gra,tym = b.get_new_batch()
# print("acc shape", len(acc))
# print("gra shape", len(gra))
# print("gyr shape", len(gyr))
# print(tym,"\n")