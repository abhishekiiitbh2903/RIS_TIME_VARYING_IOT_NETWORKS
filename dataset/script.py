import numpy as np 
import tensorflow as tf
import scipy.io 


def prepare_dataset(train_ori_path, val_ori_path, test_ori_path):
    try:
        mat = scipy.io.loadmat('train_ori_path', verify_compressed_data_integrity=False)
        train_ori = mat['phasebit']
        train_ori = np.reshape(train_ori, (train_ori.shape[0], train_ori.shape[1], 1))
        train_ori = tf.dtypes.cast(train_ori, tf.float32)
        print("Training Data loaded and processed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        mat = scipy.io.loadmat('val_ori_path', verify_compressed_data_integrity=False)
        val_ori = mat['phasebit']
        val_ori = np.reshape(val_ori, (val_ori.shape[0], val_ori.shape[1], 1))
        val_ori = tf.dtypes.cast(val_ori, tf.float32)
        print("Validation Data loaded and processed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


    try:
        mat = scipy.io.loadmat('test_ori_path', verify_compressed_data_integrity=False)
        test_ori = mat['phasebit']
        test_ori = np.reshape(test_ori, (test_ori.shape[0], test_ori.shape[1], 1))
        test_ori = tf.dtypes.cast(test_ori, tf.float32)
        print("Test Data loaded and processed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        file_info = scipy.io.whosmat('train_ori_path')
        print("File info:", file_info)
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        file_info = scipy.io.whosmat('val_ori_path')
        print("File info:", file_info)
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        file_info = scipy.io.whosmat('test_ori_path')
        print("File info:", file_info)
    except Exception as e:
        print(f"An error occurred: {e}")

    return train_ori, val_ori, test_ori