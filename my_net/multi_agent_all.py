#-*- coding: UTF-8 -*-
import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']='3'
import tensorflow as tf
#import env
import tf2_our_env_new as env #we add this file
#import a3c
import our_a3c as a3c
import load_trace
import shutil

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.06
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 7
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100

M_IN_K = 1000.0
MILLISECONDS_IN_SECOND = 1000.0
FEEDBACK_DURATION = 1000.0  #in milisec
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = '../out/results'
LOG_FILE = '../out/results/log'
TEST_LOG_FOLDER = '../data/traces/cooked_test_results/'
TRAIN_TRACES = '../data/traces/cooked_traces/'

NN_MODEL = None
#NN_MODEL = './results/nn_model_ep_78300.ckpt'

#add
FRAME_RATE = 25

MIN_BIT_RATE = 500
MAX_BIT_RATE = 4400
DEFAULT_ACTION = 3
DEFAULT_BIT_RATE = 1000
DEFAULT_VMAF = 40
MODIFY_BIT_RATE = ['xp',-400,-100,0,100,200,400]
BIT_RATE_INTERVAL = 100

def compute_reward(vmaf, last_vmaf, rebuf_size, delay,bitrate, lastbitrate):

    REBUF_PENALTY_LOW = 2
    REBUF_PENALTY_HIGH = 6 #4、6
    SMOOTH_FACTOR = 0.5
    DELAY_PENALTY_LOW = 4
    DELAY_PENALTY_HIGH = 12
    VMAF_TARGET = 85
    BITRATE_FACTOR = 2.5
    VMAF_FACTOR_LOW = 2.5
    VMAF_FACTOR_HIGH = 4

    if rebuf_size <= 0.15:
        REBUF_PENALTY = REBUF_PENALTY_LOW
    else:
        REBUF_PENALTY = REBUF_PENALTY_HIGH

    if delay <= 0.25:
        DELAY_PENALTY = DELAY_PENALTY_LOW
    else:
        DELAY_PENALTY = DELAY_PENALTY_HIGH

    if vmaf <= VMAF_TARGET:
        VMAF_FACTOR = VMAF_FACTOR_LOW
    else:
        VMAF_FACTOR = VMAF_FACTOR_HIGH

    reward = 10 - VMAF_FACTOR * np.abs((vmaf - VMAF_TARGET) / 5) - REBUF_PENALTY * rebuf_size - np.abs((vmaf - last_vmaf) / 10) - DELAY_PENALTY * delay
    # reward = 10 - np.abs((vmaf - VMAF_TARGET)/10) - REBUF_PENALTY * rebuf_size - SMOOTH_FACTOR * np.abs((bitrate - lastbitrate)/200) - DELAY_PENALTY * delay - BITRATE_FACTOR * (bitrate - MIN_BIT_RATE) / MAX_BIT_RATE
    # reward = 10 - 1.2 * np.abs((vmaf - VMAF_TARGET) / 10) - REBUF_PENALTY * rebuf_size - SMOOTH_FACTOR * np.abs(
    #     (bitrate - lastbitrate) / 200) - DELAY_PENALTY * delay

    return reward

def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    # run test script
    os.system('python rl_test_all.py ' + nn_model)
    
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()


def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session(config=config) as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)


        summary_ops, summary_vars = a3c.build_summaries()
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        maxreward = -1000
        sess.run(tf.global_variables_initializer())
        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
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
            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                print(len(s_batch))
                print(len(s_batch[0]))
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
                total_batch_len += len(r_batch)
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
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Min_reward: ' + str(min_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: min_reward,
                summary_vars[3]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                if(epoch % 10000 == 0):
                    maxreward = 0
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                testing(epoch, 
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)

                if(avg_reward >= maxreward ):
                    maxreward = avg_reward
                    os.system('cp ' + SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt.* ./goodmodels/")


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.Session(config=config) as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        action = DEFAULT_ACTION
        bitrate = DEFAULT_BIT_RATE
        lastbitrate = DEFAULT_BIT_RATE
        vmaf = DEFAULT_VMAF
        lastvmaf = DEFAULT_VMAF
        bitrateid = (bitrate - MIN_BIT_RATE)/BIT_RATE_INTERVAL

        last_recv_bitrate = MIN_BIT_RATE #kbps
        recv_bitrate = MIN_BIT_RATE #kbps

        action_vec = np.zeros(A_DIM)
        action_vec[action] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        rand_times = 1
        while True:  # experience video streaming forever
            # the action is from the last decision
            # this is to make the framework similar to the real
 
            bitrateid = (bitrate - MIN_BIT_RATE)/BIT_RATE_INTERVAL

            delay, wait_time, buffer_size, packet_loss_rate, \
            received_bit_rate, nack_sent_count, end_of_video,  rebuf_size, real_frame_data_size, real_frame_data_vmaf, real_frame_data_ti, real_bandwidth,file = \
                net_env.get_video_chunk(bitrateid)

            delay = delay / MILLISECONDS_IN_SECOND

            time_stamp += wait_time
            recv_bitrate = received_bit_rate #kbps

            vmaf = real_frame_data_vmaf
            ti = real_frame_data_ti
            reward = compute_reward(vmaf,lastvmaf,rebuf_size,delay,bitrate,lastbitrate)

            r_batch.append(reward)

            lastvmaf = vmaf
            lastbitrate = bitrate
            last_recv_bitrate = recv_bitrate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)


            # dequeue history record
            state = np.roll(state, -1, axis=1)

 
            if(delay < 1e-4):
                delay = 1e-4
            if(packet_loss_rate < 1e-4):
                packet_loss_rate = 1e-4
            # this should be S_INFO number of terms
            state[0, -1] = 2 * (bitrate  - (MIN_BIT_RATE + MAX_BIT_RATE)/2) / float(MAX_BIT_RATE - MIN_BIT_RATE)  # last quality, kilo bits / s
            state[1, -1] = 2 * (rebuf_size / (FEEDBACK_DURATION / MILLISECONDS_IN_SECOND) - 0.5)
            state[2, -1] = 2 * (received_bit_rate - (MAX_BIT_RATE + MIN_BIT_RATE)/2) / float(MAX_BIT_RATE - MIN_BIT_RATE)  # kilo bits / s
            state[3, -1] = 2 * (np.log10(delay)/4 + 0.5) # 1 sec
            state[4, -1] = 2 * (np.log10(packet_loss_rate)/4 + 0.5)  # packet_loss_rate
            state[5, -1] = np.log10(float(nack_sent_count)+1) - 1  #number of nack sent

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)

            hitcount = np.zeros(A_DIM)
            for i in range(rand_times):
                hit = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                hitcount[hit] = hitcount[hit] + 1 
            action = hitcount.argmax()

            #print(action_prob)
            #print(max(action_prob[0][1:])-min(action_prob[0][1:]))

            #action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            Xtemp = 'X1.0'

            if(MODIFY_BIT_RATE[action] == 'xp'):
                if(packet_loss_rate <= 1e-4):
                    bitrate = int(np.floor(float((1 - packet_loss_rate + 1e-4) * bitrate)/float(BIT_RATE_INTERVAL))*BIT_RATE_INTERVAL)
                    Xtemp = 'X' + str(float(1 - packet_loss_rate + 1e-4))
                else:
                    bitrate = int(np.floor(float((1 - packet_loss_rate - 0.1) * bitrate)/float(BIT_RATE_INTERVAL))*BIT_RATE_INTERVAL)
                    Xtemp = 'X' + str(float(1 - packet_loss_rate - 0.1))
            else:
                bitrate = MODIFY_BIT_RATE[action] + bitrate


            #hitcount = np.zeros(A_DIM)
            #for i in range(rand_times):
            #    hit = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            #    hitcount[hit] = hitcount[hit] + 1 
            #action = hitcount.argmax()

            #bitrate = MODIFY_BIT_RATE[action] + bitrate
            if(bitrate < MIN_BIT_RATE):
                bitrate = MIN_BIT_RATE
            if(bitrate > MAX_BIT_RATE):
                bitrate = MAX_BIT_RATE

            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            #if(int(time_stamp / MILLISECONDS_IN_SECOND)%10000 < 100):
            if(MODIFY_BIT_RATE[action] == 'xp'):
                log_file.write(str(time_stamp / MILLISECONDS_IN_SECOND) + '\t' +
                           '('+ Xtemp + ')\t' +
                           str(bitrate) + '\t' +
                           str(real_frame_data_size) + '\t' +
                           str(received_bit_rate) + '\t' +
                           str(delay) + '\t' +
                           str(rebuf_size) + '\t' +
                           str(packet_loss_rate) + '\t' +
                           str(nack_sent_count) + '\t' +
                           str(ti) + '\t' +
                           str(vmaf) + '\t' +
                           str(real_bandwidth) + '\t' +
                           str(reward) + '\t' +
                           str(file) + '\n')
            else:
                log_file.write(str(time_stamp / MILLISECONDS_IN_SECOND) + '\t' +
                           '('+ str(MODIFY_BIT_RATE[action]) + ')\t' +
                           str(bitrate) + '\t' +
                           str(real_frame_data_size) + '\t' +
                           str(received_bit_rate) + '\t' +
                           str(delay) + '\t' +
                           str(rebuf_size) + '\t' +
                           str(packet_loss_rate) + '\t' +
                           str(nack_sent_count) + '\t' +
                           str(ti) + '\t' +
                           str(vmaf) + '\t' +
                           str(real_bandwidth) + '\t' +
                           str(reward) + '\t' +
                           str(file) + '\n')
            log_file.flush()


            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                #startbit = np.random.uniform(-6.0,6.0)
                #if(startbit < 1.0):
                #    startbit = 1.0
                #if(startbit > 25.0):
                #    startbit = 25.0
            
                #np.random.randint(MIN_BIT_RATE/BIT_RATE_INTERVAL,MAX_BIT_RATE/BIT_RATE_INTERVAL + 1)*BIT_RATE_INTERVAL
                last_bit_rate = DEFAULT_BIT_RATE
                bitrate = DEFAULT_BIT_RATE  # use the default action here
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

    shutil.rmtree(SUMMARY_DIR)
    os.mkdir(SUMMARY_DIR)
    shutil.rmtree('../out/goodmodels')
    os.mkdir('../out/goodmodels')

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    print(11111111)
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()
    print(22222222)
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    print(33333333)
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
