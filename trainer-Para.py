import subprocess
import random
import time
import pickle
from keras.models import load_model
import numpy as np
import tensorflow as tf
import numpy as np

map_settings = [32, 40, 48, 56, 64]

def create_programs(game_number, epsilon):
    programs = []

    seed = np.random.randint(0, 16000000)
    if game_number % 50 == 0:
        programs.append('halite.exe -s {} --no-logs --no-timeout --width {} --height {} "python DQN-PER-Para1.py {}" "python DQN-PER-Para0.py {}"'.format(seed, map_settings[seed%5], map_settings[seed%5], epsilon, epsilon))
        programs.append('halite.exe -s {} --no-logs --no-replay --no-timeout --width {} --height {} "python DQN-PER-Para2.py {}" "python DQN-PER-Para3.py {}"'.format(seed//2, map_settings[(seed+1)%5], map_settings[(seed+1)%5], epsilon, epsilon))
        programs.append('halite.exe -s {} --no-logs --no-replay --no-timeout --width {} --height {} "python DQN-PER-Para4.py {}" "python DQN-PER-Para5.py {}"'.format(seed//3, map_settings[(seed+2)%5], map_settings[(seed+2)%5], epsilon, epsilon))
    else:
        if game_number % 3 == 0:
            programs.append('halite.exe -s {} --no-logs --no-replay --no-timeout --width {} --height {} "python DQN-PER-Para1.py {}" "python DQN-PER-Para0.py {}"'.format(seed, map_settings[seed%5], map_settings[seed%5], epsilon, epsilon))
            programs.append('halite.exe -s {} --no-logs --no-replay --no-timeout --width {} --height {} "python DQN-PER-Para2.py {}" "python DQN-PER-Para3.py {}"'.format(seed//2, map_settings[(seed+1)%5], map_settings[(seed+1)%5], epsilon, epsilon))
            programs.append('halite.exe -s {} --no-logs --no-replay --no-timeout --width {} --height {} "python DQN-PER-Para4.py {}" "python DQN-PER-Para5.py {}"'.format(seed//3, map_settings[(seed+2)%5], map_settings[(seed+2)%5], epsilon, epsilon))
        else:
            programs.append('halite.exe -s {} --no-logs --no-replay --no-timeout --width {} --height {} "python DQN-PER-Para0.py {}" "python DQN-PER-Para1.py {}" "python DQN-PER-Para2.py {}" "python DQN-PER-Para3.py {}"'.format(seed, map_settings[seed%5], map_settings[seed%5], epsilon, epsilon, epsilon, epsilon))
            programs.append('halite.exe -s {} --no-logs --no-replay --no-timeout --width {} --height {} "python DQN-PER-Para4.py {}" "python DQN-PER-Para5.py {}"'.format(seed//3, map_settings[(seed+2)%5], map_settings[(seed+2)%5], epsilon, epsilon))

    return programs

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
epsilon = 1
epsilon_decay = 0.9988
epsilon_min = 0.03
start = 2001
train_step = 1300
epsilon *= epsilon_decay**start
model = load_model('my_model.h5', custom_objects={'huber_loss': huber_loss, 'tf': tf})
np.random.seed()


for i in range(start, 15000):
    print("\n=========== GAME NUMBER : " + str(i) + '\n')

    programs = create_programs(i, epsilon)
    processes = [subprocess.Popen(program) for program in programs]
    # wait
    for process in processes:
        process.wait()

    epsilon *= epsilon_decay
    epsilon = max(epsilon_min, epsilon)

    print("Training...")
    buffer0 = pickle.load(open("buffer0.data", "rb"))
    buffer1 = pickle.load(open("buffer1.data", "rb"))
    buffer2 = pickle.load(open("buffer2.data", "rb"))
    buffer3 = pickle.load(open("buffer3.data", "rb"))
    buffer4 = pickle.load(open("buffer4.data", "rb"))
    buffer5 = pickle.load(open("buffer5.data", "rb"))

    #Train le DQN
    for _ in range (train_step):
        if _ %100 ==0:
            print(_)
        batch0 = buffer0.sample(32)
        batch_states0 = np.array([i[0] for i in batch0])
        batch_target0 = np.array([i[1] for i in batch0])
        batch1 = buffer1.sample(32)
        batch_states1 = np.array([i[0] for i in batch1])
        batch_target1 = np.array([i[1] for i in batch1])

        batch2 = buffer2.sample(32)
        batch_states2 = np.array([i[0] for i in batch2])
        batch_target2 = np.array([i[1] for i in batch2])
        batch3 = buffer3.sample(32)
        batch_states3 = np.array([i[0] for i in batch3])
        batch_target3 = np.array([i[1] for i in batch3])

        batch4 = buffer4.sample(32)
        batch_states4 = np.array([i[0] for i in batch4])
        batch_target4 = np.array([i[1] for i in batch4])
        batch5 = buffer5.sample(32)
        batch_states5 = np.array([i[0] for i in batch5])
        batch_target5 = np.array([i[1] for i in batch5])

        model.fit(np.concatenate((batch_states0, batch_states1, batch_states2, batch_states3, batch_states4, batch_states5), axis=0), np.concatenate((batch_target0, batch_target1, batch_target2, batch_target3, batch_target4, batch_target5), axis=0), epochs=1, verbose=0, batch_size=192)

    model.save('my_model.h5')
    if i >= 600 and i%100 == 0:
        model.save('my_model{}.h5'.format(i))
