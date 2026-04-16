import face_recognition
import numpy as np

def get_face_data(image_array):
    """
    获取图像中的人脸位置和特征编码 (128维)
    :param image_array: numpy array 格式的 RGB 图像
    :return: (locations, encodings)
    """
    # 检测人脸位置 (top, right, bottom, left)
    face_locations = face_recognition.face_locations(image_array)
    # 提取特征编码
    face_encodings = face_recognition.face_encodings(image_array, face_locations)
    
    return face_locations, face_encodings

def recognize_faces(unknown_encodings, known_encodings_dict, tolerance=0.5):
    """
    将未知人脸特征与已知人脸库进行比对
    :param unknown_encodings: 待识别的人脸特征列表
    :param known_encodings_dict: 已知人脸字典 {name: encoding}
    :param tolerance: 容错率，越低越严格 (默认0.6，这里设0.5以减少误判)
    :return: 识别出的名字列表
    """
    known_names = list(known_encodings_dict.keys())
    known_encs = list(known_encodings_dict.values())
    
    names = []
    for face_encoding in unknown_encodings:
        if not known_encs:
            names.append("Unknown")
            continue
            
        # 比对所有人脸
        matches = face_recognition.compare_faces(known_encs, face_encoding, tolerance=tolerance)
        name = "Unknown"

        # 使用距离最小（最相似）的那个人脸
        face_distances = face_recognition.face_distance(known_encs, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                
        names.append(name)
    return names
