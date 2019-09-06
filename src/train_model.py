import pandas as pd
import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix, find

import src.config as config
from src.rec_mf import RecMF
from src.util_recsys import *


# ----------------------------------------------------------------------------------------------------------------------
# Limiting GPU memory growth
#
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# ----------------------------------------------------------------------------------------------------------------------
# Load Dataset
#
def load_matrix_csv(path):

    df = pd.read_csv(path, sep=',', header=None)

    user_size = max(df[0].values) + 1
    item_size = max(df[1].values) + 1

    ret_matrix = csr_matrix((df[2].values, (df[0].values, df[1].values)), shape=(user_size, item_size))

    return df, ret_matrix


_, mat_train = load_matrix_csv(config.train_path)
_, mat_valid = load_matrix_csv(config.valid_path)
_, mat_test = load_matrix_csv(config.test_path)

mat_train /= config.quantize_unit
mat_valid /= config.quantize_unit
mat_test /= config.quantize_unit

num_users = mat_train.shape[0]
num_items = mat_train.shape[1]


# ----------------------------------------------------------------------------------------------------------------------
# Model Declaration, including training setting
#
rec_model = RecMF(num_users, num_items, config.dim_factors, config.max_quantize)

optimizer = tf.optimizers.Adam(config.learning_rate)
ema = tf.train.ExponentialMovingAverage(decay=config.emv_decay)
binary_ce = tf.keras.losses.BinaryCrossentropy()

# ----------------------------------------------------------------------------------------------------------------------
# Model Training
#
steps_per_epoch = int(num_users * num_items / config.batch_size)
max_step = config.max_epoch * steps_per_epoch


def sample_data():
    # Sample user-item index
    sampled_u_index = np.random.randint(num_users, size=config.batch_size)
    sampled_i_index = np.random.randint(num_items, size=config.batch_size)
    sampled_value = map(lambda x: mat_train[sampled_u_index[x], sampled_i_index[x]], range(len(sampled_u_index)))
    sampled_value = np.fromiter(sampled_value, dtype=np.float)

    # Convert entry value into one-hot sequence
    y = np.zeros([config.batch_size, config.max_quantize + 1], dtype=np.bool)
    for i in range(config.max_quantize + 1):
        y[:, i] = sampled_value > i
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # One-hot the user and item index respectively
    x = tf.concat([tf.one_hot(sampled_u_index, num_users, dtype=tf.float32),
                   tf.one_hot(sampled_i_index, num_items, dtype=tf.float32)], axis=1)
    x = tf.tile(tf.expand_dims(x, axis=1), [1, config.max_quantize, 1])

    return x, y


def model_optimization(x, y):
    with tf.GradientTape() as tape:
        # Model forwarding
        pred = rec_model(x)

        # Loss function declaration
        loss = binary_ce(y_true=y, y_pred=pred)

    gradients = tape.gradient(loss, rec_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, rec_model.trainable_variables))

    ema.apply(rec_model.trainable_variables)


def training():
    x, y = sample_data()
    model_optimization(x, y)


def testing():
    x, y = sample_data()
    pred = rec_model(x)
    return binary_ce(y_true=y, y_pred=pred)


for step in range(max_step):
    # Model training
    training()

    if step % 100 == 0:
        loss = testing()

        # Validation
        valid_user_idx, _, _ = find(mat_valid)
        valid_user_idx = set(valid_user_idx)
        valid_results = evaluate(mat_test, mat_train, rec_model, valid_user_idx, config.top_ks, use_prec_n_recl=True)

        # Testing
        test_user_idx, _, _ = find(mat_test)
        test_user_idx = set(test_user_idx) | valid_user_idx
        test_results = evaluate(mat_test + mat_valid, mat_train, rec_model, test_user_idx, config.top_ks, use_prec_n_recl=True)



        print("Step: %d / %d  testing loss = %f" % (step, steps_per_epoch, loss))






