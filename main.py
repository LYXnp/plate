# -*- coding: utf-8 -*-
import os

import config
import data_processing
import draw
import plate_process


def Detection(path):
    img = data_processing.img_process(path)
    # 车牌识别并打印记录信息
    result_list = plate_process.car_plate_reg(img)
    for i in range(0, len(result_list)):
        plate = result_list[i]['plate_no']
        plate_color = result_list[i]['plate_color']
        print("车牌信息" + str(i + 1) + "：" + plate + "  颜色：" + plate_color)
        data_processing.w_log(path, plate, plate_color)
    draw.draw_detection_box(img, result_list)


def run(path):
    flag = 0
    # 遍历文件夹
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(
                ".PNG"):
            print("图片名称：" + filename)
            # 拼接完整的文件路径
            image_path = os.path.join(path, filename)
            # 调用检测函数
            Detection(image_path)
            flag = flag + 1
            print("本次一共检测了" + str(flag) + "图片")
            print('\n')


if __name__ == '__main__':
    run(config.folder_path)
