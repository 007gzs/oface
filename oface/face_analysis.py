# encoding: utf-8
from __future__ import absolute_import, unicode_literals

import os

import numpy as np

from oface.model import SCRFD, ArcFace
from oface.core import face_align


class Face:
    def __init__(self, *, bbox, det_score, landmark, feature, sim_face_ids, extend=None):
        """
        :param bbox: 脸部范围
        :param landmark: 关键点位置
        :param det_score: 检测分数
        :param feature: 特征
        :param sim_face_ids: 相似人脸
        :param extend: 扩展数据
        """
        self.bbox = bbox
        self.det_score = det_score
        self.landmark = landmark
        self.feature = feature
        self.sim_face_ids = sim_face_ids
        self.extend = extend

    @classmethod
    def compute_sim(cls, face1, face2):
        feature1 = face1.feature if isinstance(face1, Face) else face1
        feature2 = face2.feature if isinstance(face2, Face) else face2
        return np.dot(feature1, feature2)


class FaceAnalysis:
    def __init__(
            self,
            *,
            model_path,
            name="antelope",
            det_model_name='scrfd_10g_bnkps',
            rec_model_name='glint360k_cosface_r100_fp16_0.1'
    ):
        """
        :param model_path: 模型路径
        :param det_model_name: 人脸检测模型名称
        :param rec_model_name: 人脸识别模型名称

        """
        self.det_model = SCRFD(os.path.join(model_path, name, "%s.onnx" % det_model_name))
        if rec_model_name is not None:
            self.rec_model = ArcFace(os.path.join(model_path, name, "%s.onnx" % rec_model_name))
        else:
            self.rec_model = None
        self.registered_faces = list()

    def register_face(self, face_id, face):
        """
        注册人脸

        :param face_id: 唯一标识
        :param face: Face 或 Face.feature
        """
        self.registered_faces.append((face_id, face))

    def check_face(self, face, min_sim=0.6, max_count=1):
        """

        :param face: Face
        :param min_sim: 相似度下限
        :param max_count: 返回数量
        :return:
        """
        ret = list()
        for face_id, reg_face in self.registered_faces:
            sim = Face.compute_sim(face, reg_face)
            if sim > min_sim:
                ret.append((face_id, sim))
        ret = list(sorted(ret, key=lambda x: -x[1]))
        if max_count > 0:
            return ret[:max_count]
        else:
            return ret

    def get_faces(
            self,
            image,
            *,
            img_scaled=1.0,
            max_num=0,
            dec_threshold=0.5,
            get_feature=True,
            min_sim=0.6,
            match_num=1,
            input_size=(640, 640)
    ):
        """

        :param image: 图片
        :param img_scaled: 图片已缩放比例（返回缩放前坐标）
        :param max_num: 最大返回人脸数（0为全部）
        :param dec_threshold: 人脸检测阈值
        :param get_feature: 是否返回人脸识别相关参数
        :param min_sim: 人脸识别相似度下限
        :param match_num: 人脸识别匹配返回数量
        :param input_size: 检测时人脸大小
        """
        dets, landmarks = self.det_model.detect(
            image, threshold=dec_threshold, max_num=max_num, metric='default', input_size=input_size
        )
        ret = list()
        if dets.shape[0] == 0:
            return ret

        for i in range(dets.shape[0]):
            bbox = dets[i, 0:4]
            det_score = dets[i, 4]
            landmark = None
            if landmarks is not None:
                landmark = landmarks[i]
            feature = None
            sim_face_ids = None
            extend = dict()
            if get_feature and self.rec_model is not None:
                cropped_image = self.get_cropped_image(image, landmark)
                feature, extend = self.get_feature(cropped_image)
                sim_face_ids = self.get_sim_faces(feature, min_sim, match_num)
            ret.append(Face(
                bbox=(bbox / img_scaled).astype(np.int).tolist(),
                det_score=float(det_score),
                landmark=(landmark / img_scaled).astype(np.int).tolist() if landmark is not None else None,
                feature=feature,
                sim_face_ids=sim_face_ids,
                extend=extend
            ))
        return ret

    def get_cropped_image(self, image, landmark):
        """
        获取五官对齐后图片

        :param image: 图片
        :param landmark: 五官特征点
        """
        assert landmark is not None
        return face_align.norm_crop(image, landmark=landmark)

    def get_feature(self, cropped_image):
        """
        通过五官对其后图片获取图片特征值

        :param cropped_image: 五官对齐后图片
        """
        assert self.rec_model is not None
        extend = dict()
        feat = self.rec_model.get(cropped_image)
        embedding = feat.flatten()
        embedding_norm = np.linalg.norm(embedding)
        normed_embedding = embedding / embedding_norm
        extend['feat'] = feat
        extend['cropped_image'] = cropped_image
        extend['embedding'] = embedding
        extend['embedding_norm'] = embedding_norm
        extend['normed_embedding'] = normed_embedding
        feature = normed_embedding
        return feature, extend

    def get_sim_faces(self, feature, min_sim, match_num):
        """
        通过特征值获取相似人脸

        :param feature: 特征值
        :param min_sim: 人脸识别相似度下限
        :param match_num: 人脸识别匹配返回数量
        """
        return self.check_face(feature, min_sim=min_sim, max_count=match_num)
