# Qwen TTS Deployment Project

将通义千问2.5（Qwen-2.5）模型优化并部署到移动设备上的TTS（文本转语音）项目。该项目使用0.5B版本模型，通过一系列优化技术，实现在MTK芯片的移动设备上的高效运行。

## 项目特点

- 自动下载并配置Qwen-2.5-0.5B模型
- 支持自定义训练数据集（默认提供金刚经释义测试数据集）
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
export ANDROID_HOME=~/Android/Sdk # 请修改为你的Android SDK路径
# or
export ANDROID_HOME=C:\Users\Weiyi\AppData\Local\Android\Sdk 
# 设置Android NDK路径
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk
or 
export ANDROID_NDK_HOME=C:\Users\Weiyi\AppData\Local\Android\Sdk\ndk

```

## 项目结构
```
qwen_mobile/
├── config/                    # 配置文件
│   ├── model_config.py       # 模型配置
│   └── env_config.py         # 环境配置
├── scripts/                   # 脚本目录
│   ├── download/             # 模型下载相关
│   ├── data/                 # 数据处理相关
│   ├── optimize/             # 模型优化相关
│   ├── deploy/               # 部署相关
│   └── android/              # Android项目生成相关
├── utils/                    # 工具函数
│   ├── logger_config.py      # 日志配置
│   ├── exceptions.py         # 异常定义
│   └── helpers.py           # 辅助函数
├── tests/                    # 测试文件
├── models/                   # 模型文件
│   ├── original/            # 原始模型
│   ├── pruned/             # 剪枝后模型
│   ├── quantized/          # 量化后模型
│   └── android/            # Android版本模型
├── data/                    # 数据目录
│   └── tts_dataset/        # TTS数据集
├── outputs/                 # 输出目录
├── logs/                    # 日志目录
└── android/                 # Android项目目录
```

## 快速开始

1. 环境配置
```bash
# 克隆项目
git clone https://github.com/guiyu/QwenLocalLLM.git
cd qwen_mobile

# 创建虚拟环境
# 创建一个名为qwen_env的新环境，使用Python 3.9
conda create -n qwen_env python=3.9
# 激活新环境
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
# 仅下载模型
python main.py --action download

# 仅训练模型
python main.py --action train

# 仅优化模型
python main.py --action optimize

# 仅部署到Android
python main.py --action deploy
```

## 详细说明

### 1. 数据集
默认提供金刚经释义的测试数据集，包含10段经文释义。可以通过修改 `script/data/dataset_utils.py` 来使用自己的数据集。

### 2. 模型优化
采用了多项优化技术：
- 结构化剪枝
- 动态量化
- KV Cache优化
- MTK芯片特定优化

### 3. Android部署
生成的Android应用支持：
- 实时TTS转换
- 离线运行
- 低延迟响应
- 自适应音频播放

## 常见问题

1. **为什么选择0.5B版本的模型？**
   - 考虑移动设备的硬件限制和实际需求
   - 在性能和资源占用间取得平衡
   - 提供更快的响应速度和更低的资源占用

2. **如何处理模型精度损失？**
   - 采用量化感知训练
   - 针对关键层保留更高精度
   - 实施自适应量化策略

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

本项目采用 MIT 许可证。详细信息请参见 LICENSE 文件。

## 致谢

- 感谢Qwen团队提供基础模型
- 感谢所有贡献者的支持

## 联系方式

- 提交 Issue
- 发送邮件至：[weiyi415@hotmail.com]
