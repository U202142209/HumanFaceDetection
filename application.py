from io import BytesIO

import gradio as gr
import requests
from PIL import Image, ImageDraw
from supervision import Detections
from ultralytics import YOLO

# 加载模型
model = YOLO("model.pt")


def recognize_from_url(url):
    if not str(url).startswith("http") :
        gr.Warning("请输入有效的图片URL")
        return None, None, "# 请输入有效的图片URL"
    return detect_faces(image_url=url)


def recognize_from_file(file):
    if not file:
        gr.Warning("请上传图片")
        return None, None, "# 请上传图片"
    return detect_faces(image_file=file)


def detect_faces(image_url=None, image_file=None):
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = image_file  # Image.open(image_file)
    # 获取模型的识别结果
    output = model(image)
    results = Detections.from_ultralytics(output[0])

    # 获取所有人脸矩形框的边界
    faces = []
    x_min = float('inf')
    y_min = float('inf')
    x_max = 0
    y_max = 0
    for result in results.xyxy:
        faces.append(result)
        x_min = min(x_min, result[0])
        y_min = min(y_min, result[1])
        x_max = max(x_max, result[2])
        y_max = max(y_max, result[3])

    # 稍微扩大边界
    padding = 20
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    # 确保边界不超出原图像的尺寸
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.width)
    y_max = min(y_max, image.height)

    # 裁剪含有人脸的局部图片
    face_area = image.crop((x_min, y_min, x_max, y_max)) if faces else None
    # 再原来的图片中是使用矩形框标记人脸的位置
    draw = ImageDraw.Draw(image)
    for result in results.xyxy:
        draw.rectangle(result, outline="red", width=5)
        # 创建Markdown格式的表格来展示人脸矩形框的坐标信息
    markdown_table = f"# 识别到了 {len(faces)} 张人脸\n"
    if faces:
        markdown_table += "| x_min | y_min | x_max | y_max |\n"
        markdown_table += "|-------|-------|-------|-------|\n"
        for face in faces:
            markdown_table += f"| {face[0]} | {face[1]} | {face[2]} | {face[3]} |\n"
    else:
        markdown_table = "# No faces detected."
    # 不要返回ImageDraw对象，而是返回经过绘制的Image对象
    return image, face_area, markdown_table


with gr.Blocks() as dome:
    with gr.Row():
        with gr.Column():
            # 处理输入
            gr.Markdown(
                """
                # 人脸检测模型
                ## 输入
                包含人脸图片的URL地址或者上传一张图片
                ## 输出
                - 将人脸的位置使用矩形框标记
                - 将包含人脸的区域局部显示
                - 输出图片中人脸的位置信息
                """)
            # 输入
            with gr.Tab("方式一：输入图片的URL"):
                input_url = gr.Textbox(label="Image URL", placeholder="请输入图片的URL")
                # 示例
                with gr.Accordion("Examples|示例", open=True):
                    with gr.Row():
                        example_urls = [{
                            "text": "示例1",
                            "value": "https://ts3.cn.mm.bing.net/th?id=OIP-C.vCy7mZT9tASvvQJrijoezwHaE8&w=306&h=204&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2"
                        }, {
                            "text": "示例2",
                            "value": "https://ts2.cn.mm.bing.net/th?id=OIP-C.vhQuW4FvUPVVRbs9k0DKoAHaE7&w=306&h=204&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2"
                        }, {
                            "text": "示例3",
                            "value": "https://tse1-mm.cn.bing.net/th/id/OIP-C.IoZcIcxbVXkHcvGO2MwS3wHaE8?w=287&h=191&c=7&r=0&o=5&dpr=1.3&pid=1.7"
                        }, {
                            "text": "示例4",
                            "value": "https://tse3-mm.cn.bing.net/th/id/OIP-C.OS2WBs7wIwAe1ZnOFgwpYgAAAA?w=268&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7"
                        }, ]
                        for item in example_urls:
                            gr.Button(
                                value=item["text"], size="sm",
                            ).click(
                                fn=lambda x=item["value"]: gr.Textbox(value=x, label="URL"),
                                inputs=[],
                                outputs=[input_url]
                            )
                # 按钮
                btn_url = gr.Button(value="检测人脸", variant="primary")
            with gr.Tab("方式二：上传图片"):
                input_file = gr.Image(label="Upload Image", type="pil")
                btn_img = gr.Button(value="检测人脸", variant="primary")
        with gr.Column():
            # 处理输出
            output_origin = gr.Image(type="pil", label="Original Image", show_label=False)
            output_rect = gr.Image(type="pil", label="Face Area", show_label=False, )
            output_table = gr.Markdown(label="Face Coordinates")
    # 绑定点击事件
    btn_url.click(
        fn=recognize_from_url,
        inputs=[input_url],
        outputs=[output_origin, output_rect, output_table])
    btn_img.click(
        fn=recognize_from_file,
        inputs=[input_file],
        outputs=[output_origin, output_rect, output_table])

dome.launch()
