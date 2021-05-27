###########
OFace
###########

.. image:: https://img.shields.io/pypi/v/oface.svg
       :target: https://pypi.org/project/oface

安装与升级
==========


为了简化安装过程，推荐使用 pip 进行安装

.. code-block:: bash

    pip install oface

升级 OFace 到新版本::

    pip install -U oface

如果需要安装 GitHub 上的最新代码::

    pip install https://github.com/007gzs/oface/archive/master.zip


预训练模型下载地址
=======================

`百度云下载 <https://pan.baidu.com/s/1Lp3H3oz8O6w1wC9S7CSL-w>`_ 提取码：face

`DropBox下载 <https://www.dropbox.com/sh/yhlrgfgolphqqt5/AADBiAFlVL8TYne-4L6_udCha>`_

快速使用
==========


注册+识别::

    import oface
    import cv2

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")
    face_analysis = oface.FaceAnalysis(model_path=MODEL_PATH)
    for i in range(5):
        image = cv2.imread("label%d.jpg" % i)
        faces = face_analysis.get_faces(image, max_num=1)
        if faces:
            face_analysis.register_face("label%d" % i, faces[0].feature)
    image = cv2.imread("test.jpg")
    faces = face_analysis.get_faces(image)
    res = [
        {
            'bbox': face.bbox,
            'det_score': face.det_score,
            'landmark': face.landmark,
            'sim_face_ids': [{'face_id': face_id, 'sim': float(sim)} for face_id, sim in face.sim_face_ids or []]
        }
        for face in faces
    ]
    print(res)
