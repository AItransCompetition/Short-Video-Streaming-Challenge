import sys, os
sys.path.append('./simulator/')
import argparse
import random
import numpy as np
from simulator import controller as env, short_video_load_trace

parser = argparse.ArgumentParser()
parser.add_argument('--quickstart', type=str, default='', help='Is testing quickstart')
parser.add_argument('--baseline', type=str, default='', help='Is testing baseline')
parser.add_argument('--solution', type=str, default='./', help='The relative path of your file dir, default is current dir')
parser.add_argument('--trace', type=str, default='mixed', help='The network trace you are testing (fixed, high, low, medium, middle)')
args = parser.parse_args()

RANDOM_SEED = 42  # the random seed for user retention
np.random.seed(RANDOM_SEED)
seeds = np.random.randint(100, size=(7, 2))

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
SUMMARY_DIR = 'logs'
LOG_FILE = 'logs/log.txt'
log_file = None

# QoE arguments
alpha = 1
beta = 1.85
gamma = 1
theta = 0.5
ALL_VIDEO_NUM = 7
# baseline_QoE = 600  # baseline's QoE
# TOLERANCE = 0.1  # The tolerance of the QoE decrease
MIN_QOE = -1e4
all_cooked_time = []
all_cooked_bw = []

# record the last chunk(which will be played) of each video to aid the calculation of smoothness
last_chunk_bitrate = [-1, -1, -1, -1, -1, -1, -1]

# calculate the smooth penalty for an action to download:
# chunk:[chunk_id] of the video:[download_video_id] with bitrate:[quality]
def get_smooth(net_env, download_video_id, chunk_id, quality):
    if download_video_id == 0 and chunk_id == 0:  # is the first chunk of all
        return 0
    if chunk_id == 0:  # needs to find the last chunk of the last video
        last_bitrate = last_chunk_bitrate[download_video_id - 1]
    else:
        last_bitrate = net_env.players[download_video_id - net_env.get_start_video_id()].get_downloaded_bitrate()[chunk_id - 1]
    return abs(quality - VIDEO_BIT_RATE[last_bitrate])


def test(isBaseline, isQuickstart, user_id, trace_id, user_sample_id):
    global LOG_FILE
    global log_file
    if isBaseline:  # Testing baseline algorithm
        sys.path.append('./baseline/')
        if user_id == 'no_save':
            import no_save as Solution
            LOG_FILE = 'logs/log_nosave.txt'
            log_file = open(LOG_FILE, 'w')
        sys.path.remove('./baseline/')
    elif isQuickstart:  # Testing quickstart algorithm
        sys.path.append('./quickstart/')
        if user_id == 'fixed_preload':
            import fix_preload as Solution
            LOG_FILE = 'logs/log_fixpreload.txt'
            log_file = open(LOG_FILE, 'w')
        sys.path.remove('./quickstart/')        
    else:  # Testing participant's algorithm
        sys.path.append(user_id)
        import solution as Solution
        sys.path.remove(user_id)
        LOG_FILE = 'logs/log.txt'
        log_file = open(LOG_FILE, 'w')

    # start the test
    print('------------trace ', trace_id, '--------------', file=log_file)

    solution = Solution.Algorithm()
    solution.Initialize()

    # all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(trace_path)
    net_env = env.Environment(user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], ALL_VIDEO_NUM, seeds)

    # Decision variables
    download_video_id, bit_rate, sleep_time = solution.run(0, 0, 0, False, 0, net_env.players, True)  # take the first step

    assert 0 <= bit_rate <= 2, "Your chosen bitrate [" + str(bit_rate) + "] is out of range. "\
        + "\n   % Hint: you can only choose bitrate 0 - 2 %"
    assert 0 <= download_video_id <= 4, "The video you choose is not in the current Recommend Queue. \
        \n   % You can only choose the current play video and its following four videos %"

    # output the first step
    if sleep_time != 0:
        print("You choose to sleep for ", sleep_time, " ms", file=log_file)
    else:
        print("Download Video ", download_video_id, " chunk (",
              net_env.players[download_video_id].get_chunk_counter() + 1, " / ",
              net_env.players[download_video_id].get_chunk_sum(), ") with bitrate ", bit_rate,
              file=log_file)

    # sum of wasted bytes for a user
    sum_wasted_bytes = 0
    QoE = 0
    last_played_chunk = -1  # record the last played chunk
    bandwidth_usage = 0  # record total bandwidth usage

    while True:
        # calculate the quality and smooth for this download step taken
        quality = 0
        smooth = 0
        if sleep_time == 0:
            # the last chunk id that user watched
            max_watch_chunk_id = net_env.user_models[
                download_video_id - net_env.get_start_video_id()].get_watch_chunk_cnt()
            # last downloaded chunk id
            download_chunk = net_env.players[download_video_id - net_env.get_start_video_id()].get_chunk_counter()
            if max_watch_chunk_id >= download_chunk:  # the downloaded chunk will be played
                if download_chunk == max_watch_chunk_id:  # maintain the last_chunk_bitrate array
                    last_chunk_bitrate[download_video_id] = bit_rate
                quality = VIDEO_BIT_RATE[bit_rate]
                smooth = get_smooth(net_env, download_video_id, download_chunk, quality)
                print("Causing smooth penalty: ", smooth, file=log_file)


        delay, rebuf, video_size, end_of_video, \
        play_video_id, waste_bytes = net_env.buffer_management(download_video_id, bit_rate, sleep_time)
        # print(delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes)

        # Update bandwidth usage
        bandwidth_usage += video_size

        # Update bandwidth wastage
        sum_wasted_bytes += waste_bytes  # Sum up the bandwidth wastage

        # print log info of the last operation
        if play_video_id < ALL_VIDEO_NUM:
            # the operation results
            current_chunk = net_env.players[0].get_play_chunk()
            # print(current_chunk)
            current_bitrate = net_env.players[0].get_video_quality(max(int(current_chunk - 1e-10), 0))
            print("Playing Video ", play_video_id, " chunk (", current_chunk, " / ", net_env.players[0].get_chunk_sum(),
                  ") with bitrate ", current_bitrate, file=log_file)
            # if max(int(current_chunk - 1e-10), 0) == 0 or last_played_chunk == max(int(current_chunk - 1e-10), 0):
            #     # is the first chunk or the same chunk as last time(already calculated) of the current video
            #     smooth = 0
            # else:  # needs to calc smooth
            #     last_bitrate = net_env.players[0].get_video_quality(int(current_chunk - 1e-10) - 1)
            #     smooth = current_bitrate - last_bitrate
            #     if smooth == 0:
            #         print("Your bitrate is stable and smooth. ", file=log_file)
            #     else:
            #         print("Your bitrate changes from ", last_bitrate, " to ", current_bitrate, ".", file=log_file)
            # last_played_chunk = max(int(current_chunk - 1e-10), 0)
        else:
            print("Finished Playing!", file=log_file)
        if rebuf != 0:
            print("You caused rebuf for Video ", play_video_id, " of ", rebuf, " ms", file=log_file)
        print("*****************", file=log_file)


        # Update QoE:
        # qoe = alpha * VIDEO_BIT_RATE[bit_rate] \
        #           - beta * rebuf \
        #           - gamma * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate])

        one_step_QoE = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000.
        QoE += one_step_QoE
        # if rebuf != 0:
        #     print("bitrate:", VIDEO_BIT_RATE[bit_rate], "rebuf:", rebuf, "smooth:", smooth)

        if QoE < MIN_QOE:  # Prevent dead loops
            print('Your QoE is too low...(Your video seems to have stuck forever) Please check for errors!')
            return np.array([-1e9, bandwidth_usage,  QoE, sum_wasted_bytes, net_env.get_wasted_time_ratio()])

        # play over all videos
        if play_video_id >= ALL_VIDEO_NUM:
            print("The user leaves.", file=log_file)
            print("The user leaves.")
            break

        # Apply the participant's algorithm to decide the args for the next step
        download_video_id, bit_rate, sleep_time = solution.run(delay, rebuf, video_size, end_of_video, play_video_id, net_env.players, False)

        # print log info of the last operation
        print("\n\n*****************", file=log_file)
        # the operation detail
        if sleep_time != 0:
            print("You choose to sleep for ", sleep_time, " ms", file=log_file)
        else:
            print("Download Video ", download_video_id, " chunk (", net_env.players[download_video_id - play_video_id].get_chunk_counter() + 1, " / ",
                  net_env.players[download_video_id - play_video_id].get_chunk_sum(), ") with bitrate ", bit_rate, file=log_file)
    # Score
    S = QoE - theta * bandwidth_usage * 8 / 1000000.
    print("Your score is: ", S)

    # QoE
    print("Your QoE is: ", QoE)
    # wasted_bytes
    print("Your sum of wasted bytes is: ", sum_wasted_bytes)
    print("Your download/watch ratio (downloaded time / total watch time) is: ", net_env.get_wasted_time_ratio())

    # end the test
    print('------------trace ', trace_id, '--------------\n\n', file=log_file)
    return np.array([S, bandwidth_usage,  QoE, sum_wasted_bytes, net_env.get_wasted_time_ratio()])


def test_all_traces(isBaseline, isQuickstart, user_id, trace, user_sample_id):
    avg = np.zeros(5) * 1.0
    cooked_trace_folder = 'data/network_traces/' + trace + '/'
    global all_cooked_time, all_cooked_bw
    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(cooked_trace_folder)
    for i in range(len(all_cooked_time)):
        print('------------trace ', i, '--------------')
        avg += test(isBaseline, isQuickstart, user_id, i, user_sample_id)
        print('---------------------------------------\n\n')
    avg /= len(all_cooked_time)
    print("\n\nYour average indexes under [", trace, "] network is: ")
    print("Score: ", avg[0])
    print("Bandwidth Usage: ", avg[1])
    print("QoE: ", avg[2])
    print("Sum Wasted Bytes: ", avg[3])
    print("Wasted time ratio: ", avg[4])
    return avg


def test_user_samples(isBaseline, isQuickstart, user_id, trace, sample_cnt):  # test 50 user sample
    seed_for_sample = np.random.randint(10000, size=(1001, 1))
    avgs = np.zeros(5)
    for j in range(sample_cnt):
        global seeds
        np.random.seed(seed_for_sample[j])
        seeds = np.random.randint(10000, size=(7, 2))  # reset the sample random seeds
        avgs += test_all_traces(isBaseline, isQuickstart, user_id, trace, j)
    avgs /= sample_cnt
    print("Score: ", avgs[0])
    print("Bandwidth Usage: ", avgs[1])
    print("QoE: ", avgs[2])
    print("Sum Wasted Bytes: ", avgs[3])
    print("Wasted time ratio: ", avgs[4])


if __name__ == '__main__':
    assert args.trace in ["mixed", "high", "low", "medium"]
    if args.baseline == '' and args.quickstart == '':
        test_all_traces(False, False, args.solution, args.trace, 0)  # 0 means the first user sample.
    elif args.quickstart != '':
        test_all_traces(False, True, args.quickstart, args.trace, 0)
    else:
        test_all_traces(True, False, args.baseline, args.trace, 0)
