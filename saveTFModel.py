#!/usr/bin/env python
#coding=utf-8
"""
@version: 1.0
@author: jin.dai
@license: Apache Licence
@contact: daijing491@gmail.com
@software: PyCharm
@file: saveTFModel.py
@time: 2017/7/29 17:30
@description:
"""

import tensorflow as tf
import numpy as np

def main():
    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = 4*x+4

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.zeros([1]))
    y_predict = tf.matmul(w,x)+b

    loss = tf.reduce_mean(tf.square(y-y_predict))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    isTrain = False
    train_steps = 100
    checkpoint_steps = 50
    checkpoint_dir = 'temp/'

    saver = tf.train.Saver()
    x_data = np.reshape(np.random.rand(10).astype(np.float32), (10,1))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        if isTrain:
            for i in range(train_steps):
                sess.run(train, feed_dict={x:x_data})
                if (i+1) % checkpoint_steps == 0:
                    save_path = saver.save(sess, checkpoint_dir+"model.ckpt", global_step=i+1)
                    print("save to", save_path)
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("restore from", ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                pass
            print(sess.run(w))
            print(sess.run(b))


if __name__ == '__main__':
    main()