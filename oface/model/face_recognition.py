# encoding: utf-8
from __future__ import absolute_import, unicode_literals

import cv2
import onnx
import numpy as np

from .base import ONNXModel


__all__ = ['FaceRecognition', 'ArcFace']


class FaceRecognition(ONNXModel):
    def __init__(self, model_file=None, session=None):
        super().__init__(model_file=model_file, session=session, task_name='recognition')

    def get(self, image):
        raise NotImplementedError


class ArcFace(FaceRecognition):
    def __init__(self, model_file=None, session=None):
        super().__init__(model_file=model_file, session=session)
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        assert input_shape[2] == 112 and input_shape[3] == 112
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1

    def get(self, image):
        assert image.shape[2] == 3
        input_size = tuple(image.shape[0:2][::-1])
        assert input_size == self.input_size
        blob = cv2.dnn.blobFromImage(
            image, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        feat = net_outs[0]
        return feat
