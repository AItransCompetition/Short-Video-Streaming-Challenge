# import numpy as np
# import fixed_env as env
# import load_trace
# import matplotlib.pyplot as plt
import itertools
from video_player import VIDEO_CHUNCK_LEN

VIDEO_BIT_RATE = [750,1200,1850]  # Kbps
BITS_IN_BYTE = 8
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
MILLISECONDS_IN_SECOND = 1000.0
# SUMMARY_DIR = './logs'
# LOG_FILE = './results/log_sim_mpc'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

# past errors in bandwidth
# past_errors = []
# past_bandwidth_ests = []

#size_video1 = [3155849, 2641256, 2410258, 2956927, 2593984, 2387850, 2554662, 2964172, 2541127, 2553367, 2641109, 2876576, 2493400, 2872793, 2304791, 2855882, 2887892, 2474922, 2828949, 2510656, 2544304, 2640123, 2737436, 2559198, 2628069, 2626736, 2809466, 2334075, 2775360, 2910246, 2486226, 2721821, 2481034, 3049381, 2589002, 2551718, 2396078, 2869088, 2589488, 2596763, 2462482, 2755802, 2673179, 2846248, 2644274, 2760316, 2310848, 2647013, 1653424]
# size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
# size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
# size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
# size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
# size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
# size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

# def get_chunk_size(quality, index):
#     if ( index < 0 or index > 48 ):
#         return 0
#     # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
#     sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
#     return sizes[quality]


def mpc(past_bandwidth, past_bandwidth_ests, past_errors, all_future_chunks_size, P, buffer_size, chunk_sum, video_chunk_remain, last_quality):
    # print("MPC:::", buffer_size, "\n")

    CHUNK_COMBO_OPTIONS = []
    # np.random.seed(RANDOM_SEED)

    # assert len(VIDEO_BIT_RATE) == A_DIM

    # all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

    # net_env = env.Environment(all_cooked_time=all_cooked_time,
                            #   all_cooked_bw=all_cooked_bw)

    # log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    # log_file = open(log_path, 'wb')

    # time_stamp = 0

    # last_bit_rate = DEFAULT_QUALITY
    # bit_rate = DEFAULT_QUALITY

    # action_vec = np.zeros(A_DIM)
    # action_vec[bit_rate] = 1

    # s_batch = [np.zeros((S_INFO, S_LEN))]
    # a_batch = [action_vec]
    # r_batch = []
    # entropy_record = []

    # video_count = 0

    # make chunk combination options
    for combo in itertools.product([0,1,2], repeat=P):
        CHUNK_COMBO_OPTIONS.append(combo)

    # while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        
        # if 

        # delay, sleep_time, buffer_size, rebuf, \
        # video_chunk_size, \
        # end_of_video, video_chunk_remain = \
        #     net_env.get_video_chunk(bit_rate)

        # time_stamp += delay  # in ms
        # time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                                    VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        # log scale reward
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # reward = BITRATE_REWARD[bit_rate] \
        #          - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[last_bit_rate])


        # r_batch.append(reward)

        # last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        # log_file.write((str(time_stamp / M_IN_K) + '\t' +
        #                str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
        #                str(buffer_size) + '\t' +
        #                str(rebuf) + '\t' +
        #                str(video_chunk_size) + '\t' +
        #                str(delay) + '\t' +
        #                str(reward) + '\n').encode())
        # log_file.flush()

        # retrieve previous state
        # if len(s_batch) == 0:
        #     state = [np.zeros((S_INFO, S_LEN))]
        # else:
        #     state = np.array(s_batch[-1], copy=True)

        # # dequeue history record
        # state = np.roll(state, -1, axis=1)

        # # this should be S_INFO number of terms
        # state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        # state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        # state[2, -1] = rebuf
        # state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        # state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

    # ================== MPC =========================
    # shouldn't change the value of past_bandwidth_ests and past_errors in MPC
    copy_past_bandwidth_ests = past_bandwidth_ests
    # print("past bandwidth ests: ", copy_past_bandwidth_ests)
    copy_past_errors = past_errors
    # print("past_errs: ", copy_past_errors)
    
    curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
    if ( len(copy_past_bandwidth_ests) > 0 ):
        curr_error = abs(copy_past_bandwidth_ests[-1]-past_bandwidth[-1])/float(past_bandwidth[-1])
    copy_past_errors.append(curr_error)

    # pick bitrate according to MPC           
    # first get harmonic mean of last 5 bandwidths
    past_bandwidths = past_bandwidth[-5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]
    #if ( len(state) < 5 ):
    #    past_bandwidths = state[3,-len(state):]
    #else:
    #    past_bandwidths = state[3,-5:]
    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += (1/float(past_val))
    harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))
    # print("harmonic_bandwidth:", harmonic_bandwidth)

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if ( len(copy_past_errors) < 5 ):
        error_pos = -len(copy_past_errors)
    max_error = float(max(copy_past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth/(1 + max_error)  # robustMPC here
    # print("future_bd:", future_bandwidth)
    copy_past_bandwidth_ests.append(harmonic_bandwidth)

    # future chunks length (try 4 if that many remaining)
    # last_index = int(chunk_sum - video_chunk_remain)
    
    # if ( chunk_sum - last_index < 5 ):
        # future_chunk_length = chunk_sum - last_index

    # all possible combinations of 5 chunk bitrates (9^5 options)
    # iterate over list and for each, compute reward and store max reward combination
    max_reward = float('-inf')
    best_combo = ()
    start_buffer = buffer_size
    # print("start_buffer:", start_buffer)

    lys_rebuf = 0
    lys_combo = (0,0,0,0,0)
    #start = time.time()
    for combo in CHUNK_COMBO_OPTIONS:
        # combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer  # ms
        bitrate_sum = 0
        smoothness_diffs = 0
        # last_quality = int( bit_rate )
        # print(combo)
        lys_curr_buffer = []
        lys_download_time = []
        lys_download_size = []
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            # print(len(all_future_chunks_size[0]))
            # print(chunk_quality)
            # print(position)
            # index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = MILLISECONDS_IN_SECOND * (all_future_chunks_size[chunk_quality][position]/1000000.)/(future_bandwidth) # this is MB/MB/s --> seconds
            # print("download time:", MILLISECONDS_IN_SECOND, "*",  (all_future_chunks_size[chunk_quality][position]/1000000.), "/", future_bandwidth)
            #lys test
            lys_curr_buffer.append(curr_buffer)
            lys_download_time.append(download_time)
            lys_download_size.append(all_future_chunks_size[chunk_quality][position])
            if ( curr_buffer < download_time ):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += VIDEO_CHUNCK_LEN
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            # bitrate_sum += BITRATE_REWARD[chunk_quality]
            # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
        
        reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time/1000.) - (smoothness_diffs/1000.)
        # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)
        if ( reward >= max_reward ):
            if (best_combo != ()) and best_combo[0] < combo[0]:
                best_combo = combo
            else:
                best_combo = combo
            # print(combo, ": bitrate_sum: ", bitrate_sum, ", curr_rebuffer: ", curr_rebuffer_time, ", reward: ", reward)
            # print("have buffer: ", lys_curr_buffer)
            # print("download_time: ", lys_download_time)
            # print("download_size: ", lys_download_size)
            max_reward = reward
            lys_rebuf = curr_rebuffer_time
            lys_combo = combo
            # send data to html side (first chunk of best combo)
            send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
            if ( best_combo != () ): # some combo was good
                send_data = best_combo[0]

    bit_rate = send_data
    # if curr_rebuffer_time != 0:
    # print("choosing:", lys_combo, ", rebuf ", lys_rebuf)
    # print("Your expected future_bandwidth is: (B/s)", future_bandwidth)
    # print("\n")
    return bit_rate
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        # s_batch.append(state)

        # if end_of_video:
        #     log_file.write(('\n').encode())
        #     log_file.close()

        #     last_bit_rate = DEFAULT_QUALITY
        #     bit_rate = DEFAULT_QUALITY  # use the default action here

        #     del s_batch[:]
        #     del a_batch[:]
        #     del r_batch[:]

        #     action_vec = np.zeros(A_DIM)
        #     action_vec[bit_rate] = 1

        #     s_batch.append(np.zeros((S_INFO, S_LEN)))
        #     a_batch.append(action_vec)
        #     entropy_record = []

        #     print("video count", video_count)
        #     video_count += 1

        #     if video_count >= len(all_file_names):
        #         break

        #     log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
        #     log_file = open(log_path, 'wb')


# if __name__ == '__main__':
#     main()

