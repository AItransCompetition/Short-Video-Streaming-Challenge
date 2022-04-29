import os

COOKED_TRACE_FOLDER = './data/network_traces/mixed/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    cooked_time = []
    cooked_bw = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
    print('Loading traces finished')
    return cooked_time, cooked_bw


if __name__ == '__main__':
    load_trace()
