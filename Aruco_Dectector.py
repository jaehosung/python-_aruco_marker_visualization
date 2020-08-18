# %%

import glob
import numpy as np
import math
import cv2
from cv2 import aruco
import os
import matplotlib.pyplot as plt
import pandas as pd

def getVectorFromDic(idx0, idx1):
    return marker_dict[idx0]["tvec"] - marker_dict[idx1]["tvec"]

def getPositionFromDic(idx0):
    return marker_dict[idx0]["tvec"]
def getAngleFromDic(idx1, idx0, idx2):
    return getAngle(getVectorFromDic(idx2, idx0), getVectorFromDic(idx1, idx0))

def getDistanceFromDic(idx0, idx1):
    return np.linalg.norm(getVectorFromDic(idx0, idx1))

def getJointVec(arr, joint_num):
    return np.array(arr[joint_num * 4:joint_num * 4 + 3])

def normalization(vec):
    return vec / np.linalg.norm(vec)


def getSize(vec):
    return np.linalg.norm(vec)


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def getAngle(v1, v2):
    return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

def vec_norm(vec):
    return vec / np.linalg.norm(vec)

def getUpVectorFromRvecs(idx):
    rtx = cv2.Rodrigues(rvecs[idx])[0]
    return rtx.T[2]

def getCenterOnImageFromDic(idx):
    return np.average(marker_dict[idx]["corner"], axis=0)

def getCenterOnImageFromDic(idx):
    return np.average(marker_dict[idx]["corner"], axis=0)

''' 
Load cameraparameters 
'''
parameter_dir = r"C:\Users\ZAIO\dev\openpose\openpose\models\cameraParameters\flir\20200810"

camera_parameters = {}
# camera_idxs = ["18284509","18284511","18284512"]
camera_idxs = ["18284509"]
parameter_types = ["CameraMatrix", "Intrinsics", "Distortion"]

for camera_idx in camera_idxs:
    fileToLoad = os.path.join(parameter_dir, camera_idx + ".xml")
    fs = cv2.FileStorage(fileToLoad, cv2.FILE_STORAGE_READ)
    _camera_parameter = {}
    for parameter_type in parameter_types:
        _camera_parameter[parameter_type] = fs.getNode(parameter_type).mat()
        camera_parameters[camera_idx] = _camera_parameter

# 3d points to image (2nd image)
extri = [0] * 3
intri = [0] * 3
dist = [0] * 3

for i in range(len(camera_idxs)):
    rotation_matrix = camera_parameters[camera_idxs[i]][parameter_types[0]][:, 0:3]
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    tvec = camera_parameters[camera_idxs[i]][parameter_types[0]][:, 3]

    extri[i] = camera_parameters[camera_idxs[i]][parameter_types[0]]
    intri[i] = camera_parameters[camera_idxs[i]][parameter_types[1]]
    dist[i] = camera_parameters[camera_idxs[i]][parameter_types[2]]

camera_index = 0

mtx = camera_parameters[camera_idxs[camera_index]][parameter_types[1]]
dist = camera_parameters[camera_idxs[camera_index]][parameter_types[2]]


''' 
Load files 
Detect markers
'''

imagefiles_dir = r'I:\20200814_Last_test4\cam01\90'
os.mkdir(os.path.join(imagefiles_dir,"processed"))
imagefiles_dir_png = os.path.join(imagefiles_dir,"*.png")
imagefiles = glob.glob(imagefiles_dir_png)
print(imagefiles)

data_dic = {}

for image_data in imagefiles:
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    length_of_axis = 0.09
    markerLength = 0.18
    frame = cv2.imread(image_data, cv2.IMREAD_UNCHANGED)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    rvecs, tvecs, objPts = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
    imaxis = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    if rvecs is not None and len(rvecs) == 4:
        for i in range(len(rvecs)):
            imaxis = aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)

        # Marker data to dictionary
        marker_dict = {}

        for idx, marker_index in enumerate(ids):
            marker_index = marker_index[0]
            _vecs = {}
            _vecs["rvec"] = rvecs[idx][0]
            _vecs["tvec"] = tvecs[idx][0]
            _vecs["corner"] = corners[idx][0]
            marker_dict[marker_index] = _vecs
        # print("tvecs and rvecs for each marker\n")
        # pp.pprint(marker_dict)

        plt.figure()
        plt.figure(figsize=(15, 15))

        data_array = []
        for idx0, idx1 in [[0, 1], [1, 2], [2, 3]]:
            cen_x, cen_y = np.average([getCenterOnImageFromDic(idx0), getCenterOnImageFromDic(idx1)], axis=0)
            distance = getDistanceFromDic(idx0, idx1)
            text = plt.text(int(cen_x), int(cen_y), "{:.3f}m".format(distance), fontsize=9, color="red")
            data_array.append(distance)
            text.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        plt.imshow(imaxis)
        dir, file = os.path.split(image_data)
        new_dir = os.path.join(dir, "processed")
        print(os.path.join(new_dir, file))
        plt.savefig(os.path.join(new_dir, file), dpi=300)
        data_dic[file] = data_array
        # plt.show()

data_result = pd.DataFrame.from_dict(data_dic, orient='index',columns=['0-1', '1-2', '2-3'])
data_result.to_csv(os.path.join(imagefiles_dir,"result.csv"))