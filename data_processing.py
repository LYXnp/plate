import copy
import config
import cv2
import numpy as np
from datetime import datetime
import csv
import os


def decodePlate(reds):  # 识别后处理
    pre = 0
    newPress = []
    for i in range(len(reds)):
        if reds[i] != 0 and reds[i] != pre:
            newPress.append(reds[i])
        pre = reds[i]
    plate = ""
    for i in newPress:
        plate += config.plateName[int(i)]
    return plate


def rec_pre_precessing(img, size=(48, 168)):  # 识别前处理
    img = cv2.resize(img, (168, 48))
    img = img.astype(np.float32)
    img = (img / 255 - config.mean_value) / config.std_value  # 归一化 减均值 除标准差
    img = img.transpose(2, 0, 1)  # h,w,c 转为 c,h,w
    img = img.reshape(1, *img.shape)  # channel,height,width转为batch,channel,height,channel
    return img


def get_plate_result(img, session_rec):  # 识别后处理
    img = rec_pre_precessing(img)
    y_onnx_plate, y_onnx_color = session_rec.run([session_rec.get_outputs()[0].name, session_rec.get_outputs()[1].name],
                                                 {session_rec.get_inputs()[0].name: img})
    index = np.argmax(y_onnx_plate, axis=-1)
    index_color = np.argmax(y_onnx_color)
    plate_color = config.plate_color_list[index_color]
    # print(y_onnx[0])
    plate_no = decodePlate(index[0])
    return plate_no, plate_color


def get_split_merge(img):  # 双层车牌进行分割后识别
    h, w, c = img.shape
    img_upper = img[0:int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h):, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img


def order_points(pts):  # 关键点排列 按照（左上，右上，右下，左下）的顺序排列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):  # 透视变换得到矫正后的图像，方便识别
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def my_letter_box(img, size=(640, 640)):  #
    h, w, c = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)

    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top


def xywh2xyxy(boxes):  # xauth坐标变为 左上 ，右下坐标 x1,y1  x2,y2
    xauth = copy.deepcopy(boxes)
    xauth[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xauth[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xauth[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xauth[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xauth


def my_nms(boxes, iou_thresh):  # nms
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep


def restore_box(boxes, r, left, top):  # 返回原图上面的坐标
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top

    boxes[:, [0, 2, 5, 7, 9, 11]] /= r
    boxes[:, [1, 3, 6, 8, 10, 12]] /= r
    return boxes


def detect_pre_precessing(img, img_size):  # 检测前处理
    img, r, left, top = my_letter_box(img, img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
    img = img / 255
    img = img.reshape(1, *img.shape)
    return img, r, left, top


def post_precessing(diets, r, left, top, conf_thresh=0.3, iou_thresh=0.5):  # 检测后处理
    choice = diets[:, :, 4] > conf_thresh
    diets = diets[choice]
    diets[:, 13:15] *= diets[:, 4:5]
    box = diets[:, :4]
    boxes = xywh2xyxy(box)
    score = np.max(diets[:, 13:15], axis=-1, keepdims=True)
    index = np.argmax(diets[:, 13:15], axis=-1).reshape(-1, 1)
    output = np.concatenate((boxes, score, diets[:, 5:13], index), axis=1)
    reserve_ = my_nms(output, iou_thresh)
    output = output[reserve_]
    output = restore_box(output, r, left, top)
    return output


def img_process(path):
    with open(path, 'rb') as f:
        img_decode = f.read()
    img_np_ = np.frombuffer(img_decode, np.uint8)
    img = cv2.imdecode(img_np_, cv2.COLOR_RGB2BGR)  # 转为opencv格式
    return img


def w_log(path, plate, plate_color):
    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    plate = plate[:2]+'.'+plate[2:]
    data = [time_str, path, plate, plate_color]
    with open('log.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)
