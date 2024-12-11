# 文件路径: main.py

import argparse
import logging
from pathlib import Path
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MobileLLMPipeline:
    def __init__(self):
        self.project_root = Path(__file__).parent.resolve()
        self.model_dir = self.project_root / "models"
        self.android_dir = self.project_root / "android"
        
        # 创建必要的目录
        for dir_path in [
            self.model_dir / "original",
            self.model_dir / "quantized",
            self.model_dir / "android",
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_model(self):
        """下载模型"""
        logger.info("Downloading model...")
        from scripts.download_model import download_phi2_model
        
        try:
            model_path = self.model_dir / "original" / "phi-2"
            success = download_phi2_model(model_path)
            if not success:
                raise Exception("Model download failed")
            logger.info("Model downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False
    
    def optimize_model(self):
        """优化模型"""
        logger.info("Optimizing model...")
        from scripts.optimize_model import optimize_model
        
        try:
            input_path = self.model_dir / "original" / "phi-2"
            output_path = self.model_dir / "quantized" / "model_quantized.onnx"
            success = optimize_model(input_path, output_path)
            if not success:
                raise Exception("Model optimization failed")
            logger.info("Model optimized successfully")
            return True
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return False
    
    def deploy_android(self):
        """部署到Android"""
        logger.info("Deploying to Android...")
        from scripts.deploy_android import deploy_to_android
        
        try:
            model_path = self.model_dir / "quantized" / "model_quantized.onnx"
            success = deploy_to_android(model_path, self.android_dir)
            if not success:
                raise Exception("Android deployment failed")
            logger.info("Android deployment successful")
            return True
        except Exception as e:
            logger.error(f"Android deployment failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Mobile LLM Deployment Tool")
    parser.add_argument(
        '--action',
        choices=['all', 'download', 'optimize', 'deploy'],
        default='all',
        help='Action to perform'
    )
    args = parser.parse_args()
    
    pipeline = MobileLLMPipeline()
    
    try:
        if args.action in ['all', 'download']:
            if not pipeline.download_model():
                return 1
                
        if args.action in ['all', 'optimize']:
            if not pipeline.optimize_model():
                return 1
                
        if args.action in ['all', 'deploy']:
            if not pipeline.deploy_android():
                return 1
                
        logger.info("Pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())