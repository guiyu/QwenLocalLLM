# 文件路径: script/deploy/deployment_manager.py
# 新建文件

import logging
import os
from pathlib import Path
import subprocess
import shutil

logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self, config):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        
        # 确保所有必要的目录存在
        self._init_directories()
    
    def _init_directories(self):
        """初始化所有必要的目录"""
        dirs = [
            self.project_root / "models" / "original",
            self.project_root / "models" / "pruned",
            self.project_root / "models" / "quantized",
            self.project_root / "models" / "android",
            self.project_root / "data" / "tts_dataset",
            self.project_root / "android"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self):
        """运行完整的部署流程"""
        try:
            # 1. 下载模型
            logger.info("Step 1: Downloading model...")
            if not self._run_script("scripts/run_download.py"):
                raise Exception("Model download failed")
            
            # 2. 生成数据集
            logger.info("Step 2: Generating dataset...")
            if not self._run_script("scripts/prepare_dataset.py"):
                raise Exception("Dataset preparation failed")
            
            # 3. 训练/微调模型
            logger.info("Step 3: Fine-tuning model...")
            if not self._run_script("scripts/train_model.py"):
                raise Exception("Model fine-tuning failed")
            
            # 4. 优化模型
            logger.info("Step 4: Optimizing model...")
            if not self._run_script("scripts/optimize_model.py"):
                raise Exception("Model optimization failed")
            
            # 5. 部署到Android
            logger.info("Step 5: Deploying to Android...")
            if not self._run_script("scripts/deploy_android.py"):
                raise Exception("Android deployment failed")
            
            # 6. 构建Android项目
            logger.info("Step 6: Building Android project...")
            if not self._build_android_project():
                raise Exception("Android build failed")
            
            logger.info("Deployment pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {str(e)}")
            return False
    
    def _run_script(self, script_path):
        """运行Python脚本"""
        try:
            script_path = self.project_root / script_path
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Script execution failed: {str(e)}")
            return False
    
    def _build_android_project(self):
        """构建Android项目"""
        try:
            android_dir = self.project_root / "android"
            
            # 检查是否存在gradlew
            if not (android_dir / "gradlew").exists():
                logger.error("gradlew not found in Android project")
                return False
            
            # 运行构建
            gradlew_path = str(android_dir / "gradlew")
            os.chmod(gradlew_path, 0o755)  # 添加执行权限
            
            result = subprocess.run(
                [gradlew_path, "assembleDebug"],
                cwd=str(android_dir),
                check=True
            )
            
            if result.returncode == 0:
                # 复制APK到输出目录
                outputs_dir = self.project_root / "outputs"
                outputs_dir.mkdir(exist_ok=True)
                
                apk_path = android_dir / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
                shutil.copy2(str(apk_path), str(outputs_dir / "qwen_tts.apk"))
                
                logger.info(f"APK generated at: {outputs_dir}/qwen_tts.apk")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Android build failed: {str(e)}")
            return False