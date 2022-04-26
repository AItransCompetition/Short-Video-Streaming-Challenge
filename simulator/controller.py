# input: download_video_id, bitrate, sleep_time
# output: info needed by schedule algorithm
# buffer: ms

import numpy as np
import math
from numpy.lib.utils import _split_line
from video_player import Player, VIDEO_CHUNCK_LEN
from user_module import Retention
from network_module import Network

USER_FILE = 'logs/sample_user/user.txt'
# user_file = open(USER_FILE, 'wb')
LOG_FILE = 'logs/log.txt'
log_file = open(LOG_FILE, 'a')

NEW = 0
DEL = 1

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
PLAYER_NUM = 5

class Environment:
    def __init__(self, user_sample_id, all_cooked_time, all_cooked_bw, video_num, seeds):
        self.players = []
        self.seeds = seeds
        global USER_FILE
        USER_FILE = 'logs/sample_user/user_'+str(user_sample_id)+'.txt'
        self.user_file = open(USER_FILE, 'wb')
        self.user_models = []  # Record the user action(Retention class) for the current video, update synchronized with players
        self.video_num = video_num
        self.video_cnt = 0
        self.play_video_id = 0
        self.network = Network(all_cooked_time, all_cooked_bw)
        self.timeline = 0.0
        # for ratio
        self.total_watched_len = 0.0
        self.total_downloaded_len = 0.0

        # print(self.watch_ratio)

        # self.download_permit = set()
        for p in range(PLAYER_NUM):
            # self.download_permit.add(p)
            self.players.append(Player(p))
            user_time, user_retent_rate = self.players[-1].get_user_model()
            self.user_models.append(Retention(user_time, user_retent_rate, seeds[self.video_cnt]))
            self.total_watched_len += self.user_models[-1].get_ret_duration()  # sum the total watch duration
            self.video_cnt += 1
            self.user_file.write((str(self.user_models[-1].get_ret_duration()) + '\n').encode())
            self.user_file.flush()
        
        self.start_video_id = 0
        self.end_video_id = PLAYER_NUM - 1

    def player_op(self, operation):
        if operation == NEW:
            # print('--------------ADD--------------')
            if self.video_cnt >= self.video_num:  # If exceed video cnt, no add
                return
            self.players.append(Player(self.video_cnt))
            # print("adding: ", self.video_num)
            user_time, user_retent_rate = self.players[-1].get_user_model()
            self.user_models.append(Retention(user_time, user_retent_rate, self.seeds[self.video_cnt]))
            self.video_cnt += 1
            self.total_watched_len += self.user_models[-1].get_ret_duration()  # sum the total watch duration
            # record the user retention rate
            # user_file.write((str(self.players[-1].get_watch_duration()) + '\n').encode())
            self.user_file.write((str(self.user_models[-1].get_ret_duration()) + '\n').encode())
            self.user_file.flush()
        else:
            # print('--------------DEL--------------')
            self.players.remove(self.players[0])
            self.user_models.remove(self.user_models[0])
    
    def get_start_video_id(self):
        return self.start_video_id

    def get_wasted_time_ratio(self):
        return self.total_downloaded_len / self.total_watched_len

    def play_videos(self, action_time):  # play for action_time from the start of current players queue
        # print("\n\nPlaying Video ", self.start_video_id)
        wasted_bw = 0
        buffer = 0

        # Continues to play if all the following conditions are satisfied:
        # 1) there's still action_time len
        # 2) the last video hasn't caused rebuf
        # 3) the video queue is not empty (will break inside the loop if its already empty)
        while buffer >= 0 and action_time > 0:
            # print("time_left:", action_time)
            # the timeline of the current video before this play step
            timeline_before_play = self.players[0].play_timeline
            # print("timeline_before_play: ", timeline_before_play)
            # the remain time length of the current video
            video_remain_time = self.user_models[0].get_ret_duration() - timeline_before_play
            # print("video_remain_time: ", video_remain_time)
            # the maximum play time of the current video
            max_play_time = min(action_time, video_remain_time)
            # print("max_play_time: ", video_remain_time)
            # timeline_after_play is the actual time when the play action ended( <=max_play_tm + before_play )
            timeline_after_play, buffer = self.players[0].video_play(max_play_time)
            # print("timeline_after_play: ", timeline_after_play)
            # the actual time length of this play action
            actual_play_time = timeline_after_play - timeline_before_play
            # print("actual_play_time: ", timeline_after_play)
            # consume the action_time
            action_time -= actual_play_time

            # if the current video has ended
            if actual_play_time == video_remain_time:
                # Output: the downloaded time length, the total time length, the watch duration
                print("\nUser stopped watching Video ", self.start_video_id, "( ", self.players[0].get_video_len(), " ms ) :")
                print("User watched for ", self.user_models[0].get_ret_duration(), " ms, you downloaded ", self.players[0].get_chunk_counter()*VIDEO_CHUNCK_LEN, " sec.")

                # Output the qualities and smooth of this video:
                smooth = 0
                video_qualities = []
                bitrate_cnt = min(math.ceil(self.players[0].get_play_chunk()), self.players[0].get_chunk_sum())
                # print(math.ceil(self.players[0].get_play_chunk()), "------")
                for i in range(1, bitrate_cnt):
                    video_qualities.append(self.players[0].get_video_quality(i-1))
                    smooth += abs(VIDEO_BIT_RATE[self.players[0].get_video_quality(i)] - VIDEO_BIT_RATE[self.players[0].get_video_quality(i-1)])
                video_qualities.append(self.players[0].get_video_quality(bitrate_cnt-1))
                print("Your downloaded bitrates are: ", video_qualities, ", therefore your smooth penalty is: ", smooth)

                # use watch duration as an arg for the calculation of wasted_bandwidth of this current video
                wasted_bw += self.players[0].bandwidth_waste(self.user_models[0])

                # Forward the queue head to the next video
                self.player_op(DEL)
                self.start_video_id += 1
                self.player_op(NEW)
                self.end_video_id += 1
                self.play_video_id += 1

            if self.play_video_id >= self.video_num:
                # if it has come to the end of the list
                print("played out!")
                break

        if buffer < 0:  # action ends because a video stuck(needs rebuffer)
            buffer = (-1) * action_time  # rebuf time is the remain action time(cause the player will stuck for this time too)
        return buffer, wasted_bw
              
    def buffer_management(self, download_video_id, bitrate, sleep_time):
        buffer = 0
        rebuf = 0
        end_of_video = False
        delay = 0
        video_size = 0
        wasted_bytes = 0

        if sleep_time > 0:
            delay = sleep_time
            buffer, wasted = self.play_videos(sleep_time)
            # Return the end flag for the current playing video
            if self.play_video_id == self.video_num:  # if user leaves
                end_of_video = True
            else:
                end_of_video = (self.players[self.play_video_id-self.start_video_id].get_remain_video_num() == 0)
        else:
            video_size = self.players[download_video_id-self.start_video_id].get_video_size(bitrate)
            # print("the actual download size is:", video_size)
            self.players[download_video_id - self.start_video_id].record_download_bitrate(bitrate)
            delay = math.floor(self.network.network_simu(video_size)+0.5)  # ms
            # print("the actual download delay is:", delay)
            # print("\n\n")
            # play_timeline, buffer = self.players[self.play_video_id - self.start_video_id].video_play(delay)
            buffer, wasted = self.play_videos(delay)
            self.total_downloaded_len += VIDEO_CHUNCK_LEN  # sum up the total downloaded time
            if download_video_id < self.start_video_id:
                # If the video has already been ended, we only accumulate the wastage
                print("Extra chunk downloaded for Video ", download_video_id,
                      " which the user already finished watching.\n")
                wasted += video_size  # Since its already fluently played, the download must be redundant
                end_of_video = True
            else:
                if self.play_video_id == self.video_num:  # if user leaves
                    end_of_video = True
                else:
                    end_of_video = self.players[download_video_id-self.start_video_id].video_download(VIDEO_CHUNCK_LEN)

        # Sum up the bandwidth wastage
        wasted_bytes += wasted
        if buffer < 0:
            rebuf = abs(buffer)

        return delay, rebuf, video_size, end_of_video, self.play_video_id, wasted_bytes
