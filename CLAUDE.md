# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DeepRule是一个基于深度学习的图表理解与数据提取系统，专门用于解析各种类型的图表（柱状图、折线图、饼图等）并从中提取结构化数据。该项目基于CornerNet架构，使用关键点检测技术来识别图表组件。

## 核心架构

**技术栈：**
- Python 3.7+, PyTorch 1.7.1+cu110, Django, OpenCV, pytesseract
- 基于CornerNet的关键点检测，自定义C++ corner pooling层
- Hourglass网络结构，多任务学习架构

**主要组件：**
- `models/` - 神经网络模型定义（CornerNetPureBar、CornerNetPurePie、CornerNetLine等）
- `RuleGroup/` - 图表解析规则引擎（Bar.py、Pie.py、LineQuiry.py等）
- `server_match/` - Django Web服务框架
- `config/` - 各模型配置文件（JSON格式）
- `models/py_utils/_cpools/` - C++实现的corner pooling层
- `db/` - COCO格式数据处理

## 常用开发命令

**环境设置：**
```bash
# 创建conda环境
conda create --name DeepRule --file DeepRule.txt
source activate DeepRule

# 编译C++组件（必需）
cd models/py_utils/_cpools/
python setup.py build_ext --inplace
cd ../../external/
make

# 安装依赖
pip install -r requirements-2023-pypi-cuda.txt
```

**模型训练：**
```bash
# 训练图表检测模型
python train_chart.py --cfg_file CornerNetPureBar --data_dir /path/to/data
```

**推理测试：**
```bash
# 单张图片测试
python test_pipeline_single.py --image_dir /path/to/image

# 批量处理
python test_pipe_type_cloud.py --image_path /path/to/images --type Bar
```

**Web服务部署：**
```bash
# 启动Django服务器
python manage.py runserver 8800
# 访问 localhost:8800
```

## 核心工作机制

**多模型架构：**
- 支持不同图表类型的专用模型（柱状图、折线图、饼图检测模型）
- 图表分类模型（CornerNetCls）用于识别图表类型

**关键点检测：**
- 使用corner pooling层检测图表关键特征点
- 柱状图：5个关键点 `[center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]`
- 折线图：数据点序列 `[d_1_x, d_1_y, …., d_n_x, d_n_y]`
- 饼图：三关键点表示扇形区域

**规则引擎：**
- 基于检测到的关键点进行结构化数据提取
- 支持多数据系列和复杂数据关系
- 处理图表元素的几何约束

**OCR集成：**
- 集成pytesseract进行文本识别
- 支持轴标签、数值、图例文本提取

## 开发注意事项

**硬件要求：**
- 必须支持CUDA的GPU
- 建议至少8GB显存用于训练

**依赖管理：**
- 使用特定版本的PyTorch (1.7.1+cu110)
- 必须编译C++扩展组件才能正常运行
- 数据格式采用COCO标准

**配置文件：**
- 模型配置位于 `config/` 目录，对应不同图表类型
- 使用JSON格式定义网络架构、训练参数等

**Web服务：**
- Django提供RESTful API接口
- 支持实时图表解析和数据提取
- 通过URL路由和视图函数处理请求