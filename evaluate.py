#!/usr/bin/env python
"""加载模型，评测在测试集上的表现"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import util


if __name__ == "__main__":
    config = util.initialize_from_env()
    model = util.get_model(config)
    with tf.Session() as session:
        model.restore(session)
        # Make sure eval mode is True if you want official conll results
        model.evaluate(session, official_stdout=True, eval_mode=True)
