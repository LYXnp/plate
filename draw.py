import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "SimHei.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_detection_box(image, result_list):
    # 在图像上绘制矩形框
    for i in range(0, len(result_list)):
        rect = result_list[i]['rect']
        plate = result_list[i]['plate_no']
        for i in range(len(rect)):
            rect[i] = int(rect[i])
        x1, y1, x2, y2 = rect
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # 绿色线条，宽度为2
        # 使用中文字体绘制文本
        image = cv2ImgAddText(image, f"{plate}", x1, y2, (255, 0, 0), 50)
    # 显示图像
    cv2.namedWindow('Car Plate Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Plate Detection', 1000, 600)
    cv2.imshow("Car Plate Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
