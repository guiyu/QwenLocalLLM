# Qwen TTS Deployment Project

[English Version](README_EN.md)
[中文版本](README.md)

将通义千问2.5（Qwen-2.5）模型优化并部署到移动设备上的TTS（文本转语音）项目。该项目使用0.5B版本模型，通过一系列优化技术，实现在MTK芯片的移动设备上的高效运行。

## 项目特点

- 自动下载并配置Qwen-2.5-0.5B模型
- 支持自定义训练数据集
- 完整的模型优化流程（剪枝、量化）
- 专门针对MTK芯片优化的移动端部署方案
- 完整的Android应用实现
- 详细的日志记录和错误处理机制

## 性能指标

在典型的MTK芯片上，模型性能表现如下：
- 模型加载时间：1-2秒
- 首次推理时间：300-500ms
- 后续推理时间：100-200ms
- 内存占用：峰值约1GB
- 存储空间需求：约500MB

## 环境要求

### 基础环境
- Python 3.9
- CUDA（可选，用于训练和优化阶段）
- Android Studio（用于移动端部署）
- MTK设备（用于测试部署）

### 环境变量
```bash
# 设置Android SDK路径
export ANDROID_HOME=~/Android/Sdk
# 或
export ANDROID_HOME=C:\Users\YourUsername\AppData\Local\Android\Sdk 

# 设置Android NDK路径
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk
# 或 
export ANDROID_NDK_HOME=C:\Users\YourUsername\AppData\Local\Android\Sdk\ndk
```

## 快速开始

1. 环境配置
```bash
# 克隆项目
git clone https://github.com/yourusername/QwenLocalLLM.git
cd qwen_mobile

# 创建虚拟环境
conda create -n qwen_env python=3.9
conda activate qwen_env

# 安装依赖
pip install -r requirements.txt
```

2. 运行完整流程
```bash
python main.py
```

3. 运行特定步骤
```bash
# 下载模型
python main.py --action download

# 训练模型
python main.py --action train

# 优化模型
python main.py --action optimize

# 部署到Android
python main.py --action deploy
```

## 优化技术

项目采用了多项优化技术：
- 结构化剪枝
- 动态量化
- KV Cache优化
- MTK芯片特定优化

## Android部署

生成的Android应用支持：
- 实时TTS转换
- 离线运行
- 低延迟响应
- 自适应音频播放

## 开发指南

### 代码规范
- 使用 black 进行代码格式化
- 遵循 PEP 8 规范
- 使用 pylint 进行代码检查

### 测试
```bash
# 运行单元测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_model.py
```

### 日志
- 日志文件位于 `logs/qwen_local_llm.log`
- 支持不同级别的日志记录
- 使用 `--debug` 参数启用调试日志

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 GNU General Public License v3.0 (GPL-3.0) 许可证。这意味着：

- 您可以自由使用、修改和分发本项目
- 如果您修改了代码，必须开源
- 使用本项目代码的项目必须也采用GPL协议
- 必须保留原始版权信息

详细信息请参见 LICENSE 文件。

### 商业使用须知

- 允许商业使用，但必须遵守GPL-3.0协议
- 必须开放源代码
- 衍生项目必须使用相同许可证
- 需要明确标注原项目信息

## 致谢

- 感谢Qwen团队提供基础模型
- 感谢所有贡献者的支持

## 联系方式

- 提交 Issue
- 发送邮件至：[dev@example.com]