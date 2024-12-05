# Qwen Mobile Deployment Project

这个项目旨在将通义千问2.5（Qwen-2.5）模型优化并部署到移动设备上。我们选择了较小的0.5B版本模型，通过一系列优化技术，使其能够在MTK芯片的移动设备上高效运行。

## 项目特点

本项目实现了以下核心功能：
- 自动下载并配置Qwen-2.5-0.5B模型
- 针对移动端的模型优化，包括量化、剪枝和编译
- 支持TTS（文本转语音）功能
- 专门针对MTK芯片进行优化
- 内存占用优化，运行时峰值约1GB
- 模型存储大小约500MB

## 性能指标

在典型的MTK芯片上，模型性能表现如下：
- 模型加载时间：1-2秒
- 首次推理时间：300-500ms
- 后续推理时间：100-200ms
- 内存占用：峰值约1GB
- 存储空间需求：约500MB

## 环境要求

开发环境：
- Python 3.9
- CUDA（可选，用于训练和优化阶段）
- Android Studio（用于移动端部署）
- MTK设备用于测试部署

## 快速开始

### 1. 环境配置

首先运行环境配置脚本：

```bash
python scripts/setup_environment.py
```

这个脚本会自动完成以下工作：
- 创建专用的Conda环境
- 配置国内镜像源以加速下载
- 安装所有必要的依赖
- 验证安装正确性

完成后，激活环境：
```bash
conda activate qwen_mobile
```

### 2. 模型下载和优化

运行模型处理流程：

```bash
# 下载模型
python scripts/download_model.py

# 模型优化（包括剪枝和量化）
python scripts/optimize_model.py
```

### 3. Android部署

将优化后的模型部署到Android设备：

1. 打开Android Studio项目：
```bash
cd android
./gradlew build
```

2. 将生成的模型文件复制到应用资源目录：
```bash
cp model_mobile_quantized.onnx android/app/src/main/assets/
```

3. 构建并运行Android应用。

## 项目结构

```
qwen_mobile/
├── scripts/
│   ├── setup_environment.py    # 环境配置脚本
│   ├── download_model.py       # 模型下载脚本
│   └── optimize_model.py       # 模型优化脚本
├── android/                    # Android项目目录
│   ├── app/
│   └── build.gradle
├── tests/                     # 测试文件
├── requirements.txt           # Python依赖
└── README.md                 # 项目文档
```

## 优化说明

本项目采用了多项优化技术来确保模型在移动端的高效运行：

1. 模型优化
   - 采用Int8量化减少模型大小
   - 使用结构化剪枝降低计算复杂度
   - 实现KV缓存优化推理速度

2. MTK芯片优化
   - 利用MTK的APU/NPU进行硬件加速
   - 使用MTK的Neuron Delegate优化性能
   - 根据具体芯片型号动态调整计算精度

3. 内存优化
   - 实现高效的内存管理机制
   - 采用流式处理减少内存占用
   - 优化模型加载策略

## 常见问题解答

**Q: 为什么选择0.5B版本的模型？**

A: 考虑到移动设备的硬件限制和实际使用需求，0.5B版本在性能和资源占用之间取得了最佳平衡。相比7B或1.8B版本，它能提供更快的响应速度和更低的资源占用，同时保持足够的模型性能。

**Q: 如何处理模型精度损失？**

A: 我们通过以下方式平衡精度和性能：
- 使用量化感知训练
- 针对关键层保留更高精度
- 实施自适应量化策略

## 开发指南

如果您想参与项目开发，请遵循以下步骤：

1. 代码风格
   - 使用black进行代码格式化
   - 遵循PEP 8规范
   - 使用pylint进行代码检查

2. 测试
   - 运行单元测试：`python -m pytest tests/`
   - 性能测试：`python scripts/benchmark.py`

3. 提交规范
   - 使用清晰的提交信息
   - 确保通过所有测试
   - 提供必要的文档更新

## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：
- 优化算法改进
- 性能优化
- 文档完善
- 问题报告和修复

请在提交Pull Request前阅读我们的贡献指南。


## 联系方式

如果您有任何问题或建议，请通过以下方式联系我们：
- 提交 Issue
- 发送邮件至 [项目邮箱]
- 加入我们的开发者社群

## 致谢

感谢所有为本项目做出贡献的开发者，感谢Qwen团队提供优质的基础模型。