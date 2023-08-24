
import onnxruntime

folder_path = r"C:\Users\MSI-NB\Desktop\20230518"
plate_color_list = ['黑色', '蓝色', '绿色', '白色', '黄色']
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = ((0.588, 0.193))  # 识别模型均值标准差
img_size = 640
providers = ['CPUExecutionProvider']
img_size = (img_size, img_size)
session_detect = onnxruntime.InferenceSession(r'C:\Users\MSI-NB\Desktop\plate_rec\Mymodel\plate_detect.onnx', providers=providers)   #车牌检测
session_rec = onnxruntime.InferenceSession(r'C:\Users\MSI-NB\Desktop\plate_rec\Mymodel\plate_rec_color.onnx', providers=providers)   #车牌识别