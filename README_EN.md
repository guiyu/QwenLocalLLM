# Qwen TTS Deployment Project

[English Version](README_EN.md)
[中文版本](README.md)

A project to optimize and deploy the Qwen-2.5 model for mobile devices with TTS (Text-to-Speech) capabilities. This project uses the 0.5B version model and implements efficient running on MTK chip mobile devices through a series of optimization techniques.

## Features

- Automatic download and configuration of Qwen-2.5-0.5B model
- Support for custom training datasets
- Complete model optimization pipeline (pruning, quantization)
- Mobile deployment solution specifically optimized for MTK chips
- Complete Android application implementation
- Detailed logging and error handling mechanism

## Performance Metrics

On typical MTK chips, the model performs as follows:
- Model loading time: 1-2 seconds
- Initial inference time: 300-500ms
- Subsequent inference time: 100-200ms
- Memory usage: Peak around 1GB
- Storage space requirement: About 500MB

## Prerequisites

### Basic Environment
- Python 3.9
- CUDA (optional, for training and optimization)
- Android Studio (for mobile deployment)
- MTK device (for testing deployment)

### Environment Variables
```bash
# Set Android SDK path
export ANDROID_HOME=~/Android/Sdk
# or
export ANDROID_HOME=C:\Users\YourUsername\AppData\Local\Android\Sdk 

# Set Android NDK path
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk
# or 
export ANDROID_NDK_HOME=C:\Users\YourUsername\AppData\Local\Android\Sdk\ndk
```

## Quick Start

1. Environment Setup
```bash
# Clone project
git clone https://github.com/yourusername/QwenLocalLLM.git
cd qwen_mobile

# Create virtual environment
conda create -n qwen_env python=3.9
conda activate qwen_env

# Install dependencies
pip install -r requirements.txt
```

2. Run Complete Pipeline
```bash
python main.py
```

3. Run Specific Steps
```bash
# Download model
python main.py --action download

# Train model
python main.py --action train

# Optimize model
python main.py --action optimize

# Deploy to Android
python main.py --action deploy
```

## Optimization Techniques

The project employs multiple optimization techniques:
- Structured pruning
- Dynamic quantization
- KV Cache optimization
- MTK chip-specific optimizations

## Android Deployment

The generated Android application supports:
- Real-time TTS conversion
- Offline operation
- Low-latency response
- Adaptive audio playback

## Development Guide

### Code Standards
- Use black for code formatting
- Follow PEP 8 guidelines
- Use pylint for code checking

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py
```

### Logging
- Log files located in `logs/qwen_local_llm.log`
- Support for different log levels
- Use `--debug` parameter to enable debug logging

## Contributing

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). This means:

- You are free to use, modify, and distribute this project
- If you modify the code, you must make it open source
- Projects using this code must also use the GPL license
- You must retain the original copyright notice

See the LICENSE file for details.

### Commercial Use Notice

- Commercial use is allowed, but must comply with GPL-3.0
- Source code must remain open source
- Derivative projects must use the same license
- Must clearly indicate original project information

## Acknowledgments

- Thanks to the Qwen team for providing the base model
- Thanks to all contributors

## Contact

- Submit an Issue
- Send email to: [dev@example.com]