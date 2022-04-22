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


def mpc(past_bandwidth, past_bandwidth_ests, past_errors, all_future_chunks_size, P, buffer_size, chunk_sum, video_chunk_remain, last_quality):
    # print("MPC:::", buffer_size, "\n")

    CHUNK_COMBO_OPTIONS = []

    # make chunk combination options
    for combo in itertools.product([0,1,2], repeat=P):
        CHUNK_COMBO_OPTIONS.append(combo)

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
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            # print(len(all_future_chunks_size[0]))
            # print(chunk_quality)
            # print(position)
            # index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = MILLISECONDS_IN_SECOND * (all_future_chunks_size[chunk_quality][position]/1000000.)/(future_bandwidth) # this is MB/MB/s --> seconds
            # print("download time:", MILLISECONDS_IN_SECOND, "*",  (all_future_chunks_size[chunk_quality][position]/1000000.), "/", future_bandwidth)
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
            max_reward = reward
            # send data to html side (first chunk of best combo)
            send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
            if ( best_combo != () ): # some combo was good
                send_data = best_combo[0]

    bit_rate = send_data
    return bit_rate

