# 文件路径: config/model_config.py
# 新建文件

class ModelConfig:
    # 模型配置
    MODEL_VERSION = "Qwen/Qwen2.5-0.5B"
    CACHE_DIR = "./model_cache"
    OUTPUT_DIR = "./models/original"
    
    # 训练相关配置
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    
    # 优化相关配置
    PRUNING_AMOUNT = 0.3
    QUANTIZATION_DTYPE = "int8"
    
    # 硬件相关配置
    USE_CUDA = True
    FP16_TRAINING = True

    # 添加数据集相关配置
    DATASET_DIR = "./data/tts_dataset"
    TRAIN_BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 8
    NUM_TEST_SAMPLES = 1000
    DATASET_SPLIT_RATIO = 0.9
    
    # TTS特定配置
    MEL_BINS = 80
    SAMPLE_RATE = 22050
    HOP_LENGTH = 256

    # 添加训练相关配置
    NUM_EPOCHS = 10
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    LOGGING_STEPS = 100
    SAVE_STEPS = 1000
    
    # 模型配置
    hidden_size = 768  # 需要根据实际的模型配置设置
    mel_bins = 80

     # 剪枝相关配置
    INITIAL_PRUNING_AMOUNT = 0.1
    FINAL_PRUNING_AMOUNT = 0.3
    PRUNING_STEPS = 3
    
    # 量化相关配置
    QUANTIZATION_TYPE = "dynamic"
    QUANTIZATION_BITS = 8
    ENABLE_OPTIMIZATION = True

 # Android相关配置
    ANDROID_OUTPUT_DIR = "./android"
    ANDROID_MODEL_DIR = "./models/android"
    ANDROID_MIN_SDK = 21
    ANDROID_TARGET_SDK = 31
    
    # 模型路径
    QUANTIZED_MODEL_PATH = "./models/quantized/model_quantized.onnx"
    
    # Android应用配置
    ANDROID_APP_ID = "com.example.qwentts"
    ANDROID_VERSION_CODE = 1
    ANDROID_VERSION_NAME = "1.0"
    
    # 音频配置
    AUDIO_SAMPLE_RATE = 22050
    AUDIO_CHANNELS = 1  # MONO
    AUDIO_BITS_PER_SAMPLE = 16
    
    # ONNX Runtime配置
    ONNX_RUNTIME_VERSION = "1.14.0"
    ONNX_THREAD_COUNT = 4
    ONNX_ENABLE_OPENCL = False
    
    # 性能配置
    INFERENCE_TIMEOUT = 30000  # 毫秒
    ENABLE_CPU_ARENA = True
    ENABLE_MEMORY_PATTERN = True
    ENABLE_MEMORY_OPTIMIZATION = True
    
    # 调试配置
    DEBUG_MODE = True
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"
    
    # 资源文件配置
    ASSETS_DIR_NAME = "models"
    MODEL_FILENAME = "model_quantized.onnx"
    CONFIG_FILENAME = "model_config.json"
    
    # TTS特定配置
    MAX_TEXT_LENGTH = 1000
    BATCH_SIZE = 1
    MEL_CHANNELS = 80
    
    # 错误处理配置
    MAX_RETRY_COUNT = 3
    RETRY_DELAY_MS = 1000
    
    @classmethod
    def get_android_gradle_config(cls):
        """获取Android Gradle配置"""
        return {
            "compileSdkVersion": cls.ANDROID_TARGET_SDK,
            "minSdkVersion": cls.ANDROID_MIN_SDK,
            "targetSdkVersion": cls.ANDROID_TARGET_SDK,
            "applicationId": cls.ANDROID_APP_ID,
            "versionCode": cls.ANDROID_VERSION_CODE,
            "versionName": cls.ANDROID_VERSION_NAME
        }
    
    @classmethod
    def get_model_config(cls):
        """获取模型配置"""
        return {
            "input_name": "input",
            "output_name": "output",
            "max_text_length": cls.MAX_TEXT_LENGTH,
            "mel_channels": cls.MEL_CHANNELS,
            "sample_rate": cls.AUDIO_SAMPLE_RATE,
            "quantization_bits": 8,
            "model_version": cls.ANDROID_VERSION_NAME
        }
    
    @classmethod
    def get_audio_config(cls):
        """获取音频配置"""
        return {
            "sample_rate": cls.AUDIO_SAMPLE_RATE,
            "channels": cls.AUDIO_CHANNELS,
            "bits_per_sample": cls.AUDIO_BITS_PER_SAMPLE
        }
