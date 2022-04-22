# This example aims at helping you to learn what parameters you need to decide in your algorithm.
# It only gives you clues to things you can do to improve the algorithm, so it isn't necessarily reasonable.
# You need to find a better solution to balance QoE and bandwidth waste.
# You can run this example and get results by command: python run.py --quickstart fix_preload

# Description of fixed-preload algorithm
# Fixed-preload algorithm downloads the current playing video first.
# When the current playing video download ends, it preloads the videos in the recommendation queue in order.
# The maximum of preloading size is 4 chunks for each video.
# For each preloading chunk, if possibility (using data in user_ret to esimate) > RETENTION_THRESHOLD, it is assumed that user will watch this chunk so that it should be preloaded.
# It stops when all downloads end.

# We use buffer size to decide bitrate, here is the threshold.
LOW_BITRATE_THRESHOLD = 1000
HIGH_BITRATE_THRESHOLD = 2000
# If there is no need to download, sleep for TAU time.
TAU = 500.0  # ms
# max length of PLAYER_NUM
PLAYER_NUM = 5
# user retention threshold
RETENTION_THRESHOLD = 0.65
# fixed preload chunk num
PRELOAD_CHUNK_NUM = 4

class Algorithm:
    def __init__(self):
        # fill the self params
        pass

    def Initialize(self):
        # Initialize the session or something
        pass

    # Define the algorithm here.
    # The args you can get are as follows:
    # 1. delay: the time cost of your last operation
    # 2. rebuf: the length of rebufferment
    # 3. video_size: the size of the last downloaded chunk
    # 4. end_of_video: if the last video was ended
    # 5. play_video_id: the id of the current video
    # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
    # 7. first_step: is this your first step?
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # Here we didn't make use of delay & rebuf & video_size & end_of_video.
        # You can use them or get more information from Players to help you make better decisions.

        # If it is the first step, you have no information of past steps.
        # So we return specific download_video_id & bit_rate & sleep_time.
        if first_step:
            self.sleep_time = 0
            return 0, 0, 0.0
        
        # decide download video id
        download_video_id = -1
        if Players[0].get_remain_video_num() != 0:  # downloading of the current playing video hasn't finished yet 
            download_video_id = play_video_id
        else:
            # preload videos in PLAYER_NUM one by one
            for seq in range(1, min(len(Players), PLAYER_NUM)):
                if Players[seq].get_chunk_counter() < PRELOAD_CHUNK_NUM and Players[seq].get_remain_video_num() != 0:      # preloading hasn't finished yet 
                    # calculate the possibility: P(user will watch the chunk which is going to be preloaded | user has watched from the beginning to the start_chunk)  
                    start_chunk = int(Players[seq].get_play_chunk())
                    _, user_retent_rate = Players[seq].get_user_model()
                    cond_p = float(user_retent_rate[Players[seq].get_chunk_counter()]) / float(user_retent_rate[start_chunk])
                    # if p > RETENTION_THRESHOLD, it is assumed that user will watch this chunk so that it should be preloaded.
                    if cond_p > RETENTION_THRESHOLD:
                        download_video_id = play_video_id + seq
                        break

        if download_video_id == -1:  # no need to download, sleep for TAU time
            sleep_time = TAU
            bit_rate = 0
            download_video_id = play_video_id  # the value of bit_rate and download_video_id doesn't matter
        else:
            seq = download_video_id - play_video_id
            # decide bitrate according to buffer size
            if Players[seq].get_buffer_size() > HIGH_BITRATE_THRESHOLD:
                bit_rate = 2
            elif Players[seq].get_buffer_size() > LOW_BITRATE_THRESHOLD:
                bit_rate = 1
            else:
                bit_rate = 0
            sleep_time = 0.0

        return download_video_id, bit_rate, sleep_time

