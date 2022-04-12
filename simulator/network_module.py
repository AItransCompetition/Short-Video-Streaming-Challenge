# shared by all players
MILLISECONDS_IN_SECOND = 1000.0
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0

class Network:
    def __init__(self, cooked_time, cooked_bw):
        assert len(cooked_time) == len(cooked_bw)
        
        self.cooked_time = cooked_time
        self.cooked_bw = cooked_bw
        
        # self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    # calculate the download time of a certain block
    def network_simu(self,video_chunk_size):
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE      # B/s
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time    # s
            # print("cooked_bw: ", self.cooked_bw[self.mahimahi_ptr], ", throughput: ", self.cooked_bw[self.mahimahi_ptr] / BITS_IN_BYTE )
            # print("duration")

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION  # B

            if video_chunk_counter_sent + packet_payload > video_chunk_size:  # B
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                # print("sending packet_payload: ", packet_payload, " in duration: ", fractional_time)
                delay += fractional_time    # s
                self.last_mahimahi_time += fractional_time  # s
                break

            video_chunk_counter_sent += packet_payload  # B
            # print("sending packet ", video_chunk_counter_sent, ", packet_payload ", packet_payload, ', throughput ', throughput, ", duration ", duration)
            delay += duration  # s
            # print("sending packet_payload: ", packet_payload, " in duration: ", duration)
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        # print("delay: ", delay)
        return delay  # ms