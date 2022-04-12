# Simulate the user watch pattern
import numpy as np
# import random
# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

# Rt simulate
class Retention:
    def __init__(self, user_time, user_retent_rate, watch_ratio):

        assert len(user_time) == len(user_retent_rate)

        self.user_time = user_time
        self.user_retent_rate = user_retent_rate
        video_time_len = self.user_time[-2]
        self.sample_playback_duration = watch_ratio * video_time_len

        # self.user_churn_rate = 1.0 - np.array(user_retent_rate).astype('float64')
        # self.prop = np.diff(self.user_churn_rate).ravel()

        # interval = np.random.choice(self.user_time[:-1], p=self.prop)  # ms
        # if interval == self.user_time[-2]:  # if a user proceeds to the end
        #     self.sample_playback_duration = interval
        # else:  # uniform distribute over the second
        #     self.sample_playback_duration = random.uniform(interval, interval+1000)

    def get_ret_duration(self):  # ms
        # print('sample playback duration %d' % self.sample_playback_duration)
        # print(self.sample_playback_duration)
        return self.sample_playback_duration
        
    def conditional_p(self, start_chunk, interval):
        # calculate the conditional p from chunk start_chunk
        # when user is watching the start_chunk, the p of watching chunk start_chunk+interval
        # P(X>T+1|X>T) = R(T+1) / R(T)
        cond_ret_p = float(self.user_retent_rate[start_chunk+interval]) / float(self.user_retent_rate[start_chunk])
        
        return cond_ret_p
        
        

    