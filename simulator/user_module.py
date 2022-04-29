# Simulate the user watch pattern
import numpy as np
import math
import random
VIDEO_CHUNCK_LEN = 1000.0
# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

# Rt simulate
class Retention:
    def __init__(self, user_time, user_retent_rate, seeds):

        assert len(user_time) == len(user_retent_rate)

        self.user_time = user_time
        self.user_retent_rate = user_retent_rate
        video_time_len = self.user_time[-2]

        self.user_churn_rate = 1.0 - np.array(user_retent_rate).astype('float64')
        self.prop = np.diff(self.user_churn_rate).ravel()

        np.random.seed(seeds[0])
        interval = np.random.choice(self.user_time[:-1], p=self.prop)  # ms
        if interval == self.user_time[-2]:  # if a user proceeds to the end
            self.sample_playback_duration = interval
        else:  # uniform distribute over the second
            random.seed(seeds[1])
            self.sample_playback_duration = int(random.uniform(interval, interval+1000))

    def get_ret_duration(self):  # ms
        # print('sample playback duration %d' % self.sample_playback_duration)
        # print(self.sample_playback_duration)
        return self.sample_playback_duration

    def get_watch_chunk_cnt(self):
        return math.floor(self.sample_playback_duration / VIDEO_CHUNCK_LEN)
