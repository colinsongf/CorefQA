#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

import tensorflow as tf

import util

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    config = util.initialize_from_env()

    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    model = util.get_model(config)
    saver = tf.train.Saver()

    log_dir = config["log_dir"]
    max_steps = config['num_epochs'] * config['num_docs']
    writer = tf.summary.FileWriter(log_dir, flush_secs=20)

    max_f1 = 0
    mode = 'w'

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0

        initial_step = 0
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)
            mode = 'a'
            initial_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
        fh.setFormatter(logging.Formatter(format))
        logger.addHandler(fh)

        initial_time = time.time()
        while True:
            tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
            accumulated_loss += tf_loss
            # print('tf global_step', tf_global_step)

            if tf_global_step % report_frequency == 0:
                steps_per_second = (tf_global_step - initial_step) / (time.time() - initial_time)

                average_loss = accumulated_loss / report_frequency
                logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
                writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
                accumulated_loss = 0.0

            if tf_global_step % eval_frequency == 0:
                eval_summary, eval_f1 = model.evaluate(session)

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                    util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                                         os.path.join(log_dir, "model.max.ckpt"))

                writer.add_summary(eval_summary, tf_global_step)
                writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

                logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))
                if tf_global_step > max_steps:
                    break


if __name__ == '__main__':
    main()
