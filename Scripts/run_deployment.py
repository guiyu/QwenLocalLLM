# 文件路径: scripts/run_deployment.py
# 新建文件

import sys
import logging
from pathlib import Path

# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from script.deploy.deployment_manager import DeploymentManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting deployment pipeline...")
        
        # 创建部署管理器
        manager = DeploymentManager(ModelConfig)
        
        # 运行完整部署流程
        success = manager.run_full_pipeline()
        
        if success:
            logger.info("Deployment completed successfully!")
            logger.info(f"You can find the generated APK in the outputs directory")
            return 0
        else:
            logger.error("Deployment failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error during deployment: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())