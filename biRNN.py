#!/usr/bin/env python
#coding=utf-8
"""
@version: 1.0
@author: jin.dai
@license: Apache Licence
@contact: daijing491@gmail.com
@software: PyCharm
@file: biRNN.py
@time: 2017/7/28 16:56
@description:
"""

import tensorflow as tf
import numpy as np


def main():
    # Create input data
    X = np.random.randn(2, 10, 8)

    # The second example is of length 6
    X[1,6:] = 0
    X_lengths = [10, 6]

    num_units_fw = 64
    num_units_bw = 32

    lstm_fw = tf.contrib.rnn.BasicLSTMCell(num_units=num_units_fw, forget_bias=1.0, state_is_tuple=True)
    lstm_bw = tf.contrib.rnn.BasicLSTMCell(num_units=num_units_bw, forget_bias=1.0, state_is_tuple=True)

    #cell_init_state_fw = lstm_fw.zero_state(num_units_fw, dtype=tf.float64)
    #cell_init_state_bw = lstm_bw.zero_state(num_units_bw, dtype=tf.float64)

    outputs, states  = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_fw,
        cell_bw=lstm_bw,
        dtype=tf.float64,
        sequence_length=X_lengths,
        inputs=X)

    output_fw, output_bw = outputs
    states_fw, states_bw = states

    output_all = tf.concat(outputs, 2)

    result = tf.contrib.learn.run_n(
        {"output_fw": output_fw, "output_bw": output_bw,
         "states_fw": states_fw, "states_bw": states_bw,
         "output_all": output_all},
        n=1,
        feed_dict=None)

    assert result[0]["output_fw"].shape == (2, 10, 64)
    assert (result[0]["output_fw"][1, 6, :] == np.zeros(lstm_fw.output_size)).all()
    assert (result[0]["output_bw"][1, 6, :] == np.zeros(lstm_bw.output_size)).all()
    assert (result[0]["output_all"][1, 6, :] == np.zeros(lstm_fw.output_size+lstm_bw.output_size)).all()
    print(np.sum(np.sign(np.max(np.abs(X),axis=2)),axis=1))

    print("output_fw:", result[0]["output_fw"].shape, result[0]["output_fw"].dtype)
    print("output_bw:", result[0]["output_bw"].shape, result[0]["output_bw"].dtype)
    print("states_fw:", result[0]["states_fw"].h.shape, result[0]["states_fw"].h.dtype)
    print("states_bw:", result[0]["states_bw"].h.shape, result[0]["states_bw"].h.dtype)
    print("output_all:", result[0]["output_all"].shape, result[0]["output_all"].dtype)

if __name__ == '__main__':
    main()
