from simulator.video_player import Player
class Algorithm:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0

    # Intial
    def Initialize(self):
        # Initialize your session or something
        self.buffer_size = 0

    # Define your algorithm
    # The args you can get are as follows:
    # 1. delay: the time cost of your last operation
    # 2. rebuf: the length of rebufferment
    # 3. video_size: the size of the last downloaded chunk
    # 4. end_of_video: if the last video was ended
    # 5. play_video_id: the id of the current video
    # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
    # 7. first_step: is this your first step?
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        download_video_id = 0
        bit_rate = 0
        sleep_time = 0.0
        return download_video_id, bit_rate, sleep_time
