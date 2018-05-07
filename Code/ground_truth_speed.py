import datetime as dt


TEST_FILE = '../Data/keyfreq/test/log1.txt'
CHAR_SET = {'q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m',\
          '<bcksp>','spc','<lshft>'}
BUFFER = dt.timedelta(microseconds=800000)


def find_gt_type_speed(start_time, end_time):
    count = 0
    while count < 30:
        try:
            total_clicks = 0
            clicks_start, clicks_end = None, None
            with open(TEST_FILE) as curr_log_f:
                for line in curr_log_f:
                    line_split = line.split()
                    if check_in_charset(line_split):
                        time_split = line_split[0].split(':')
                        curr_log_time = dt.datetime.fromtimestamp(float(time_split[0]))
                        curr_log_time = curr_log_time + dt.timedelta(microseconds=float(time_split[1]))
                        if start_time <= curr_log_time <= end_time:
                            total_clicks += 1
                            if clicks_start is None:
                                clicks_start = curr_log_time
                            else:
                                clicks_end = curr_log_time
                        if curr_log_time > end_time + 2 * BUFFER:
                            break
                if clicks_end is not None and clicks_start is not None:
                    return (total_clicks*60)/(clicks_end - clicks_start).total_seconds()
        except:
            print("Unable to open log file", TEST_FILE, " in try", count, " Retrying...")
        count += 1
    return 0


def check_in_charset(line_split, char_set=CHAR_SET):
    if (line_split[-1].lower() in char_set) or ("spc" in char_set and ']' in line_split[-1] and '[' in line_split[-1]):
        return True
    return False
