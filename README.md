# 人脸检测

# 简介

本次实验中，我们将会使用[YOLOV8](https://github.com/ultralytics/ultralytics)
训练[人脸数据集](https://universe.roboflow.com/large-benchmark-datasets/wider-face-ndtcz/dataset/1)
，得到一个人脸检测模型。最后使用[Gradio](https://www.gradio.app/docs)将模型部署成web服务，
实现用户输入图片的URL地址或者上传一张图片，模型识别出图片中人脸的坐标信息，并使用矩形框在原图片中标记出来。

# 前置知识

## YOLO简介

[YOLO](https://blog.csdn.net/m0_69824302/article/details/139072001)（You Only Look Once）是一个流行的开源目标检测框架，由Joseph
Redmon等人于2015年提出。

## YOLO识别原理

[YOLO识别流程](https://blog.csdn.net/weixin_42377570/article/details/128675221)

# 环境搭建

## 创建python虚拟环境

本次案例中，我的环境如下

- WWindows 10 操作系统
- Python 3.8.19
- conda 4.10.1
- pip 24.0

  如果您的环境可我不一样，可以构建python虚拟环境，这样我们后续的操作可以保证模块的版本一致，减少不必要的异常。如果您的电脑上经安装了[anaconda](https://www.anaconda.com/download/)
  ，您可以使用如下的命令创建一个python3.8的虚拟环境，
  注意，```your_env_name```是你想创建的虚拟环境的名称：

```shell
conda create -n your_env_name python=3.8
```

如果您的电脑是```Windows```操作系统，可以使用下面的命令激活你的虚拟环境：

```shell
conda activate your_env_name
```

如果你的操作系统是```linux```或者```Mac```，可以使用下面的命令激活虚拟环境：

```shell
source activate your_env_name
```

更多[Anaconda](https://anaconda.org.cn/anaconda/user-guide/getting-started/)
的常用命令，您可以[查看相关教程](https://blog.csdn.net/chenxy_bwave/article/details/119996001)。

## 安装相关的模块

### 安装pytorch

[pytorch](https://pytorch.org/)
对不同的操作系统、设备配置的安装命令不一样，需要去[pytorch官网](https://pytorch.org/)下载。根据您的操作系统选择相应的版本

例如，我的电脑系统是```windows```，```Compute Platform```的版本是 ```CUDA 11.8```，使用```conda```安装的命令如下：

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 安装ultralytics

方式一：

```shell
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
# 安装依赖
pip install -e .
```

方式二：

```shell
pip install ultralytics
```

### 测试ultralytics是否正常工作

选择一个路径，激活创建的虚拟环境，执行以下测试命令

```shell
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

如果你是第一次运行，控制台会输出正在下载模型的进度条，下载完成后，模型将进行预测，如果您看到类似于如下的信息，说明预测成功

```text
Ultralytics YOLOv8.2.60 🚀 Python-3.8.19 torch-2.3.1+cu118 CUDA:0 (NVIDIA GeForce GTX 1650 Ti, 4096MiB)
YOLOv8n summary (fused): 168 layers, 3,151,904 parameters, 0 gradients, 8.7 GFLOPs

Found https://ultralytics.com/images/bus.jpg locally at bus.jpg
image 1/1 .....your\path\bus.jpg: 640x480 4 persons, 1 bus, 1 stop sign, 63.0ms
Speed: 3.0ms preprocess, 63.0ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
Results saved to runs\detect\predict1
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

从上面的结果可以看出，模型将预测的结果储存在您当前的目录下面的

```text
runs\detect\predict1
```

打开这个文件夹，可以看到类似下面的图片

<div style="text-align:center;"><img src="http://bd.modelcube.cn/media/learning_path_image/35875/1f2c82ce943df519c14768598f8f8d2c/bus.jpg" align="center" alt="undefined" /></div>


同时，您的当前目录中会出现一个```yolov8n.pt```的文件，这是yolo从网上下载的模型权重文件，下次启动的时候就可以直接加载这个文件，无需再次下载。

# 训练模型

## 数据集简介

本次的[人脸数据集](https://universe.roboflow.com/large-benchmark-datasets/wider-face-ndtcz/dataset/1)
来源于[roboflow平台](https://universe.roboflow.com/)
，您可以将压缩包[下载到本地](https://universe.roboflow.com/large-benchmark-datasets/wider-face-ndtcz/dataset/1).
下载后解压，数据集的目录结构如下：

```text
├ People_Dataset.v1i.yolov8
├── test
├── ├── images
├── ├── ├── 111115_png.rf.256d96558873301ac68d98050cefb8ae.jpg
├── ├── ├── 111123_png.rf.77c2ba2332200f2eb7acc46355c63062.jpg
├── ├── labels
├── ├── ├── 111115_png.rf.256d96558873301ac68d98050cefb8ae.txt
├── ├── ├── 111123_png.rf.77c2ba2332200f2eb7acc46355c63062.txt
├── train
├── ├── images
├── ├── labels
├── valid
├── ├── images
├── ├── labels
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
```

这个数据集的结构符合YOLO（You Only Look Once）数据集的标准格式，用于训练YOLO模型进行对象检测。以下是数据集的各个部分和它们的作用的详细介绍：

- test:测试数据集的目录
    - test/images:包含测试图像的文件夹
    - test/labels:包含与测试图像对应的标签文件的文件夹
- train:训练数据集的目录
    - train/images:包含训练图像的文件夹
    - train/labels:包含与训练图像对应的标签文件的文件夹
- valid:验证数据集的目录
    - valid/images:包含验证图像的文件夹
    - valid/labels:包含与验证图像对应的标签文件的文件夹
- **data.yaml:(重点)** 这是一个YOLO配置文件，它定义了数据集的详细信息，如图像尺寸、类别列表、数据集的子集等。
- README.dataset.txt:这是一个文本文件，其中可能包含了关于数据集的额外说明或指导。

在YOLO的标签文件中，**每个标签文件与一个图像文件对应**，并且包含了一系列的边界框（bounding
boxes）和与之相关的类别信息。每个边界框通常由以下几部分组成：

- class_id: 对象类别ID，通常从0开始计数。
- x_center: 边界框的中心点的x坐标，相对于图像宽度归一化到0和1之间。
- y_center: 边界框的中心点的y坐标，相对于图像高度归一化到0和1之间。
- width: 边界框的宽度，相对于图像宽度归一化到0和1之间。
- height: 边界框的高度，相对于图像高度归一化到0和1之间。

这些标签文件通常以 ```.txt``` 格式保存，**并且每一行对应一个边界框**。例如，一个简单的标签文件可能如下所示：

| class_id | x_center           | y_center            | width                | height              |
|----------|--------------------|---------------------|----------------------|---------------------|
| 0        | 0.4951923076923077 | 0.2692307692307692  | 0.06129807692307692  | 0.25                |
| 0        | 0.2283653846153846 | 0.28846153846153844 | 0.03125              | 0.13341346153       |      
| 0        | 0.40625            | 0.3209134615384615  | 0.042067307692307696 | 0.20673076923076922 |
| 0        | 0.224759615384615  | 0.328125            | 0.03125              | 0.12379807692307693 |

<img src="http://bd.modelcube.cn/media/learning_path_image/35875/676c843d33b44d1526d2d7f01dcfbd19/img3.png" align="center" alt="undefined" /></div>

## 模型训练

开始训练之前，YOLOV8对项目的文件夹组织结构有一定的艳琼，我们的项目路径如下

```text 
├ YOLOV8HumanFaceDetection
├── dataset
├── ├── test
├── ├── ├── images
├── ├── ├── ├── 111115_png.rf.256d96558873301ac68d98050cefb8ae.jpg
├── ├── ├── ├── 111123_png.rf.77c2ba2332200f2eb7acc46355c63062.jpg
├── ├── ├── labels
├── ├── ├── ├── 111115_png.rf.256d96558873301ac68d98050cefb8ae.txt
├── ├── ├── ├── 111123_png.rf.77c2ba2332200f2eb7acc46355c63062.txt
├── ├── train
├── ├── ├── images
├── ├── ├── labels
├── ├── valid
├── ├── ├── images
├── ├── ├── labels
├── data.yaml
├── application.py
├── illation.py
├── README.md
├── yolov8n.pt
```

其中 ```data.yaml``` 文件的内容如下

```yaml
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 1
names: [ 'people' ]

roboflow:
  workspace: raman-hrynkevich
  project: people_dataset
  version: 1
  license: Public Domain
  url: https://universe.roboflow.com/raman-hrynkevich/people_dataset/dataset/1
```

然后我们使用下面的命令开始训练

```shell
yolo task=detect mode=train model=yolov8n.pt data="data.yaml” workers=1 epochs=50 batch=16
```

伤处命令中各个参数的解释如下：

- **task=detect:** 定任务的类型。在这里，detect 表示执行的是目标检测任务。
- **mode=train:** 指定运行模式。train 表示模型将进入训练模式。在训练模式下，模型将使用提供的数据集进行学习，并尝试优化其性能。
- **model=yolov8n.pt:** 指定训练或测试时要使用的模型文件。在这里，yolov8n.pt 是一个预训练的 YOLOv8n 模型权重文件。
- **data="data.yaml":** 指定包含数据集信息的配置文件。在这里，data.yaml 是一个 YAML 格式的文件，它应该定义了训练、验证和测试数据集的路径、类名等信息。
- **workers=1:** 指定用于数据加载的子进程数。值 1 表示使用单个工作进程。增加工作进程数可以加快数据加载的速度，但要更具电脑的性能决定
- **epochs=50:** 指定训练过程中整个数据集将被迭代多少次。
- **batch=16:** 指定每个训练批次中的样本数量。在这里，16 表示每个批次将包含 16 个样本。较大的批次大小可以提供更稳定的梯度估计，但可能需要更多的内存和计算资源。

训练完成之后，

- 项目的根目录下面会出现```runs```文件夹，它里面记录了训练过程中的日志
- 出现```yolov8n_last.pt```,储存最后一次迭代后的模型权重。
- 出现```yolov8n_best.pt```，在验证集上达到最佳性能的模型权重。

如果您的电脑配置不足，没有训练出最后的模型，我们这里提供了训练好的模型权重文件，将在下面的推理部署中使用。

# 推理部署

## 本地推理

我们提供了最终训练好的模型权重文件```model.pt```，接下来我们将使用```python```
加载这个模型文件实现推理。推理程序```illation.py```文件的内容如下：

```python
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
```

上述程序处理调用了模型，还增加了在原图中绘制出识别的人脸逻辑，最后将检测到的人脸部分保存到```face_area_detected.png```文件中。

## 使用Gradio部署

使用Gradio部署分为以下步骤：

- 模型加载
- 页面布局
- 绑定按钮事件

部署的代码在```application.py```文件中，内容如下：
但如需要的模块

```python
from io import BytesIO
import gradio as gr
import requests
from PIL import Image, ImageDraw
from supervision import Detections
from ultralytics import YOLO

# 加载模型
model = YOLO("model.pt")
```

定义函数，调用模型识别人脸

```python
def recognize_from_url(url):
    if not str(url).startswith("http"):
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
    return image, face_area, markdown_table
```

定义gradio布局，并绑定按钮事件

```python
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
```    

启动gradio服务

```python
dome.launch()
```

上述的程序实现了

- 输入：包含人脸图片的URL地址或者上传一张图片
- 输出：
    - 将人脸的位置使用矩形框标记
    - 将包含人脸的区域局部显示
    - 出图片中人脸的位置信息

核心的代码在```detect_faces```函数里面

<div style="text-align:center;"><img src="http://bd.modelcube.cn/media/learning_path_image/35875/98b122c3c88548c644210bfdd7c155df/img1.png" a<div style="text-align:center;">
<div style="text-align:center;"><img src="http://bd.modelcube.cn/media/learning_path_image/35875/d613ab6a3c6548c317c1f402fbac7a04/img2.png" align="center" alt="undefined" /></div>lign="center" alt="undefined" /></div>



