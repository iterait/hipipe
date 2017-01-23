#!/usr/bin/env python3
import argparse
import os
import tensorflow as tf
import sys

from freeze_graph import freeze_graph

class TransposeAddOneNet():

    def __init__(self, in_height, in_width):
        with tf.device('/cpu:0'):
            input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, in_height, in_width],
                                        name='input')
            var = tf.Variable(1, dtype=tf.float32, name='var')
            output_data = tf.add(tf.transpose(input_data), var, name='output')

            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

    def save(self, file_prefix):
        graph_path = '{}.graph'.format(file_prefix)
        checkpoint_path = '{}.ckpt'.format(file_prefix)
        frozen_graph_path = '{}.pb'.format(file_prefix)

        tf.train.write_graph(self._session.graph_def, '', graph_path, as_text=False)
        self._saver.save(self._session, checkpoint_path)

        freeze_graph(input_graph=graph_path,
                     input_saver="",
                     input_binary=True,
                     input_checkpoint=os.path.join(os.getcwd(), checkpoint_path),
                     output_node_names='output',
                     restore_op_name="",
                     filename_tensor_name="",
                     output_graph=os.path.join(os.getcwd(), frozen_graph_path),
                     clear_devices=True,
                     initializer_nodes="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        'Build a frozen tensorflow graph which transposes its input and adds 1 to each element. '
        'The input tensorflow placeholder is named `input` and the output is named `output`.')
    parser.add_argument('-r', '--rows', type=int,
                        required=True, help='The height for the input placeholder.')
    parser.add_argument('-c', '--cols', type=int,
                        required=True, help='The width for the input placeholder.')
    args = parser.parse_args()

    net = TransposeAddOneNet(args.rows, args.cols)
    net.save('transpose_add_one_{}_{}_net'.format(args.rows, args.cols))
