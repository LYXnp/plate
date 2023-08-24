import copy
import config
import data_processing


def rec_plate(outputs, img0, session_rec):  # 识别车牌
    dict_list = []
    for output in outputs:
        result_dict = {}
        rect = output[:4].tolist()
        land_marks = output[5:13].reshape(4, 2)
        roi_img = data_processing.four_point_transform(img0, land_marks)
        label = int(output[-1])
        score = output[4]
        if label == 1:  # 代表是双层车牌
            roi_img = data_processing.get_split_merge(roi_img)
        plate_no, plate_color = data_processing.get_plate_result(roi_img, session_rec)
        result_dict['rect'] = rect
        result_dict['landmarks'] = land_marks.tolist()
        result_dict['plate_no'] = plate_no
        result_dict['roi_height'] = roi_img.shape[0]
        result_dict['plate_color'] = plate_color
        dict_list.append(result_dict)
    return dict_list


def car_plate_reg(img):
    img0 = copy.deepcopy(img)
    img, r, left, top = data_processing.detect_pre_precessing(img, config.img_size)  # 检测前处理
    y_onnx = config.session_detect.run([config.session_detect.get_outputs()[0].name],
                                       {config.session_detect.get_inputs()[0].name: img})[0]
    outputs = data_processing.post_precessing(y_onnx, r, left, top)  # 检测后处理
    result_list = rec_plate(outputs, img0, config.session_rec)
    return result_list
