import os


COOKED_TRACE_FOLDER = './data/network_traces/middle/'
BW_ADJUST_PARA = 1


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    cooked_files.sort(key=lambda x: int(x))
    all_cooked_time = []
    all_cooked_bw = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1])*BW_ADJUST_PARA)
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)

    return all_cooked_time, all_cooked_bw
