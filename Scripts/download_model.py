import logging
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

def download_phi2_model(output_path: Path) -> bool:
    """下载Phi-2模型"""
    try:
        logger.info(f"Downloading Phi-2 model to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        model = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)
        logger.info("Phi-2 model download completed")
        return True
    except Exception as e:
        logger.error(f"Phi-2 model download failed: {e}")
        return False

def download_asr_model(output_path: Path) -> bool:
    """下载ASR模型"""
    try:
        logger.info(f"Downloading ASR model to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        # 替换模型名称为正确的公共模型
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h", trust_remote_code=True)
        model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h", trust_remote_code=True)
        
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)
        logger.info("ASR model download completed")
        return True
    except Exception as e:
        logger.error(f"ASR model download failed: {e}")
        return False

def download_tts_model(output_path: Path) -> bool:
    """下载TTS模型"""
    try:
        logger.info(f"Downloading TTS model to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/speecht5_tts", trust_remote_code=True)
        model = AutoModel.from_pretrained("microsoft/speecht5_tts", trust_remote_code=True)
        
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)
        logger.info("TTS model download completed")
        return True
    except Exception as e:
        logger.error(f"TTS model download failed: {e}")
        return False
