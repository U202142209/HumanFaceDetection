from PIL import Image, ImageDraw
from supervision import Detections
from ultralytics import YOLO

model = YOLO("model.pt")  # 加载模型
original_image = Image.open("bus.jpg")  # 加载测试图片
output = model(original_image)  # 获取模型的识别结果
results = Detections.from_ultralytics(output[0])

# 获取所有人脸矩形框的边界
x_min = float('inf')
y_min = float('inf')
x_max = 0
y_max = 0

for result in results.xyxy:
    x_min = min(x_min, result[0])
    y_min = min(y_min, result[1])
    x_max = max(x_max, result[2])
    y_max = max(y_max, result[3])

# 稍微扩大边界
padding = 20  # 你可以根据需要调整这个值
x_min -= padding
y_min -= padding
x_max += padding
y_max += padding

# 确保边界不超出原图像的尺寸
x_min = max(x_min, 0)
y_min = max(y_min, 0)
x_max = min(x_max, original_image.width)
y_max = min(y_max, original_image.height)

# 裁剪并保存含有人脸的局部图片
face_area = original_image.crop((x_min, y_min, x_max, y_max))
face_area.save("face_area_detected.png")
