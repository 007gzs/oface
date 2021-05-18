# encoding: utf-8
from __future__ import absolute_import, unicode_literals

import onnxruntime


class ONNXModel:
    def __init__(self, model_file=None, session=None, task_name=''):
        self.model_file = model_file
        self.session = session
        self.task_name = task_name
        if self.session is None:
            assert self.model_file is not None
            self.session = onnxruntime.InferenceSession(self.model_file, None)
