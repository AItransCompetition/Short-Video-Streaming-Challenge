import sys, os
sys.path.append('./simulator/')
import argparse
import random
import numpy as np
from simulator import controller as env, short_video_load_trace
import logging
import multiprocess as mp
import tensorflow as tf
__all__ = [tf]
import a3c as a3c

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--quickstart', type=str, default='', help='Is testing quickstart')
parser.add_argument('--baseline', type=str, default='', help='Is testing baseline')
parser.add_argument('--solution', type=str, default='./', help='The relative path of your file dir, default is current dir')
parser.add_argument('--trace', type=str, default='fixed', help='The network trace you are testing (fixed, high, low, medium, middle)')
args = parser.parse_args()

RANDOM_SEED = 42  # the random seed for user retention
np.random.seed(RANDOM_SEED)
seeds = np.random.randint(100, size=(7, 2))

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
SUMMARY_DIR = 'logs'
LOG_FILE = 'logs/tranin_log'

# QoE arguments
alpha = 1
beta = 1.85
gamma = 1
theta = 0.5
ALL_VIDEO_NUM = 7
# baseline_QoE = 600  # baseline's QoE
# TOLERANCE = 0.1  # The tolerance of the QoE decrease
MIN_QOE = -1e4
all_cooked_time = []
all_cooked_bw = []

# For training a3c
NUM_AGENTS = 16
S_INFO = 7
S_LEN = 8
A_DIM = 4
ACTOR_LR_RATE = 0.00025
CRITIC_LR_RATE = 0.0015
MODEL_SAVE_INTERVAL = 100
NN_MODEL = None

# log file
log_file = open(LOG_FILE+'.txt', 'w')

def reward(quality, rebuffer, smooth, price):

    rwd = alpha * quality - beta * rebuffer - gamma * smooth

    return rwd


def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central.txt',
                        filemode='a',
                        level=logging.INFO)

    with tf.Session(config=config) as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        epoch = 0
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
            parse = NN_MODEL[10:-5].split('_')
            epoch = int(parse[-1])

        maxreward = -1000
        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0
            min_reward = 0.0

            actor_gradient_batch = []
            critic_gradient_batch = []

            nreward = TRAIN_SEQ_LEN
            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                nreward = min(nreward,len(actor_gradient))

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                if(np.mean(r_batch) < min_reward):
                    min_reward = np.mean(r_batch)
                total_reward += np.mean(r_batch)
                total_td_loss += np.mean(td_batch)
                #total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.mean(info['entropy'])


            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            #mean_actor_gradient_batch = np.zeros(nreward)
            #mean_critic_gradient_batch

            actor_gradient_batch = np.divide(actor_gradient_batch , NUM_AGENTS)
            critic_gradient_batch = np.divide(critic_gradient_batch , NUM_AGENTS)
            for j in range(1,nreward):
                for i in range(NUM_AGENTS):
                    actor_gradient_batch[0][j] = np.add(actor_gradient_batch[0][j] , actor_gradient_batch[i][j])
                    critic_gradient_batch[0][j] = np.add(critic_gradient_batch[0][j] , critic_gradient_batch[i][j])

            mean_actor_gradient_batch = actor_gradient_batch[0]
            mean_critic_gradient_batch = critic_gradient_batch[0]

            actor.apply_gradients(mean_actor_gradient_batch)
            critic.apply_gradients(mean_critic_gradient_batch)

            # log training information
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_agents
            avg_entropy = total_entropy / total_agents
            '''
            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Min_reward: ' + str(min_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))
            '''
            logging.info("Epoch:%06d\tTD_loss:%6.5f\tAvg_reward:%8.2f\tMin_reward:%8.2f\tAvg_entropy:%7.6f"%\
                         (epoch,avg_td_loss,avg_reward,min_reward,avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: min_reward,
                summary_vars[3]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                print ("---------epoch %d--------" % epoch)
                if(epoch % 10000 == 0):
                    maxreward = 0
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                # testing(epoch,
                #     SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt",
                #     test_log_file)

                # if(avg_reward >= maxreward ):
                #     maxreward = avg_reward
                #     os.system('cp ' + SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt.* ./testmodel/")

            if epoch == 20000:
                sys.exit(0)


def work_agent(isBaseline, isQuickstart, user_id, trace_id, user_sample_id):

    solution = Solution.Algorithm()
    solution.Initialize()

    net_env = env.Environment(user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], ALL_VIDEO_NUM, seeds)

    with tf.Session(config=config) as sess:
        # all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(trace_path)

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)
        actor_net_params, critic_net_params = net_params_queue.get()

        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        # Decision variables
        download_video_id, bit_rate, sleep_time = solution.run(0, 0, 0, False, 0, net_env.players, True)  # take the first step
        # output the first step
        if sleep_time != 0:
            print("You choose to sleep for ", sleep_time, " ms", file=log_file)
        else:
            print("Download Video ", download_video_id, " chunk (",
                  net_env.players[download_video_id].get_chunk_counter() + 1, " / ",
                  net_env.players[download_video_id].get_chunk_sum(), ") with bitrate ", bit_rate,
                  file=log_file)

        action_vec = np.zeros(A_DIM)
        action_vec[action] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        # sum of wasted bytes for a user
        sum_wasted_bytes = 0
        QoE = 0
        last_played_chunk = -1  # record the last played chunk
        last_bitrate = -1
        bandwidth_usage = 0  # record total bandwidth usage

        time_stamp = 0
        while True:
            # calculate the quality and smooth for this download step taken
            quality = 0
            smooth = 0
            if sleep_time == 0:
                # the last chunk id that user watched
                max_watch_chunk_id = net_env.user_models[
                    download_video_id - net_env.get_start_video_id()].get_watch_chunk_cnt()
                # last downloaded chunk id
                download_chunk = net_env.players[download_video_id - net_env.get_start_video_id()].get_chunk_counter()
                if max_watch_chunk_id >= download_chunk:  # the downloaded chunk will be played
                    quality = VIDEO_BIT_RATE[bit_rate]
                    if last_bitrate != -1:  # is not the first chunk to play
                        smooth = abs(quality - VIDEO_BIT_RATE[last_bitrate])
                        # print("downloading ", download_video_id, "chunk ", download_chunk, ", bitrate switching from ", last_bitrate, " to ", bit_rate)
                    last_bitrate = bit_rate


            delay, rebuf, video_size, end_of_video, \
            play_video_id, waste_bytes = net_env.buffer_management(download_video_id, bit_rate, sleep_time)
            # print(delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes)


            # Update bandwidth usage
            bandwidth_usage += video_size

            # Update bandwidth wastage
            sum_wasted_bytes += waste_bytes  # Sum up the bandwidth wastage

            # print log info of the last operation
            if play_video_id < ALL_VIDEO_NUM:
                # the operation results
                current_chunk = net_env.players[0].get_play_chunk()
                # print(current_chunk)
                current_bitrate = net_env.players[0].get_video_quality(max(int(current_chunk - 1e-10), 0))
                print("Playing Video ", play_video_id, " chunk (", current_chunk, " / ", net_env.players[0].get_chunk_sum(),
                      ") with bitrate ", current_bitrate, file=log_file)
                # if max(int(current_chunk - 1e-10), 0) == 0 or last_played_chunk == max(int(current_chunk - 1e-10), 0):
                #     # is the first chunk or the same chunk as last time(already calculated) of the current video
                #     smooth = 0
                # else:  # needs to calc smooth
                #     last_bitrate = net_env.players[0].get_video_quality(int(current_chunk - 1e-10) - 1)
                #     smooth = current_bitrate - last_bitrate
                #     if smooth == 0:
                #         print("Your bitrate is stable and smooth. ", file=log_file)
                #     else:
                #         print("Your bitrate changes from ", last_bitrate, " to ", current_bitrate, ".", file=log_file)
                # last_played_chunk = max(int(current_chunk - 1e-10), 0)
            else:
                print("Finished Playing!", file=log_file)

            if rebuf != 0:
                print("You caused rebuf for Video ", play_video_id, " of ", rebuf, " ms", file=log_file)
            print("*****************", file=log_file)

            reward = 0
            # play over all videos
            if play_video_id >= ALL_VIDEO_NUM:
                print("The user leaves.", file=log_file)
                reward = reward(quality, rebuf, smooth)
            r_batch.append(reward)
            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)
            # dequeue history record
            state = np.roll(state, -1, axis=1)

            state[0, -1] = delay
            state[1, -1] = rebuf
            state[2, -1] = video_size
            state[3, -1] = end_of_video
            state[4, -1] = play_video_id
            state[5, -1] = net_env.players
            state[6, -1] = first_step

            # Decide the args for the next step
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            hitcount = np.zeros(A_DIM)
            for i in range(1):
                hit = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                hitcount[hit] = hitcount[hit] + 1
            action = hitcount.argmax()
            #Culculate action entropy
            entropy_record.append(a3c.compute_entropy(action_prob[0]))
            # print log info of the last operation
            print("\n\n*****************", file=log_file)
            # the operation detail
            if sleep_time != 0:
                print("You choose to sleep for ", sleep_time, " ms", file=log_file)
            else:
                print("Download Video ", download_video_id, " chunk (", net_env.players[download_video_id - play_video_id].get_chunk_counter() + 1, " / ",
                      net_env.players[download_video_id - play_video_id].get_chunk_sum(), ") with bitrate ", bit_rate, file=log_file)
            # store the state and action into batches
            if play_video_id >= ALL_VIDEO_NUM:
                #startbit = np.random.uniform(-6.0,6.0)
                #if(startbit < 1.0):
                #    startbit = 1.0
                #if(startbit > 25.0):
                #    startbit = 25.0

                #np.random.randint(MIN_BIT_RATE/BIT_RATE_INTERVAL,MAX_BIT_RATE/BIT_RATE_INTERVAL + 1)*BIT_RATE_INTERVAL
                last_qp = DEFAULT_QP
                qp = DEFAULT_QP  # use the default action here
                action = DEFAULT_ACTION

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1
                a_batch.append(action_vec)


def main():

    np.random.seed(RANDOM_SEED)
    #assert len(MODIFY_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    work_agents = []
    for i in range(NUM_AGENTS):
        work_agents.append(mp.Process(target=work_agent,
                                 args=(i, all_cooked_time, all_cooked_bw, all_file_names,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        work_agents[i].start()
    os.system('chmod -R 777 ' + SUMMARY_DIR)
    # wait unit training is done
    coordinator.join()


'''
def test_all_traces(isBaseline, isQuickstart, user_id, trace, user_sample_id):
    avg = np.zeros(5) * 1.0
    cooked_trace_folder = 'data/network_traces/' + trace + '/'
    global all_cooked_time, all_cooked_bw
    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(cooked_trace_folder)
    for i in range(len(all_cooked_time)):
        print('------------trace ', i, '--------------')
        print('------------trace ', i, '--------------', file=log_file)
        avg += test(isBaseline, isQuickstart, user_id, i, user_sample_id)
        print('------------trace ', i, '--------------\n\n', file=log_file)
        print('---------------------------------------\n\n')
    avg /= len(all_cooked_time)
    print("\n\nYour average indexes under [", trace, "] network is: ")
    print("Score: ", avg[0])
    print("Bandwidth Usage: ", avg[1])
    print("QoE: ", avg[2])
    print("Sum Wasted Bytes: ", avg[3])
    print("Wasted time ratio: ", avg[4])
    return avg


def test_user_samples(isBaseline, isQuickstart, user_id, trace, sample_cnt):  # test 50 user sample
    seed_for_sample = np.random.randint(10000, size=(1001, 1))
    avgs = np.zeros(5)
    for j in range(sample_cnt):
        global seeds
        np.random.seed(seed_for_sample[j])
        seeds = np.random.randint(10000, size=(7, 2))  # reset the sample random seeds
        avgs += test_all_traces(isBaseline, isQuickstart, user_id, trace, j)
    avgs /= sample_cnt
    print("Score: ", avgs[0])
    print("Bandwidth Usage: ", avgs[1])
    print("QoE: ", avgs[2])
    print("Sum Wasted Bytes: ", avgs[3])
    print("Wasted time ratio: ", avgs[4])
'''

if __name__ == '__main__':

    main()

    '''
    assert args.trace in ["fixed", "high", "low", "medium", "middle"]
    if args.baseline == '' and args.quickstart == '':
        test_all_traces(False, False, args.solution, args.trace, 0)  # 0 means the first user sample.
    elif args.quickstart != '':
        test_all_traces(False, True, args.quickstart, args.trace, 0)
    else:
        test_all_traces(True, False, args.baseline, args.trace, 0)
    '''
