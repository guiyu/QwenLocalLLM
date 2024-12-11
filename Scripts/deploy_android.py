# 文件路径: scripts/deploy_android.py

import logging
import shutil
from pathlib import Path
import subprocess
import os

logger = logging.getLogger(__name__)

def deploy_to_android(model_path: Path, android_dir: Path) -> bool:
    """部署到Android项目"""
    try:
        # 1. 确保Android项目目录存在
        if not android_dir.exists():
            raise Exception(f"Android project directory not found: {android_dir}")

        # 2. 复制模型文件到assets目录
        assets_dir = android_dir / "app/src/main/assets/models"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = assets_dir / "model_quantized.onnx"
        shutil.copy2(str(model_path), str(target_path))
        
        logger.info(f"Model copied to {target_path}")

        # 3. 更新local.properties
        update_local_properties(android_dir)

        # 4. 执行Gradle构建
        if not build_android_project(android_dir):
            raise Exception("Android build failed")

        # 5. 验证APK是否生成
        apk_path = android_dir / "app/build/outputs/apk/debug/app-debug.apk"
        if not apk_path.exists():
            raise Exception("APK not found")

        logger.info(f"APK generated at: {apk_path}")
        return True

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False

def update_local_properties(android_dir: Path) -> None:
    """更新local.properties文件"""
    try:
        # 获取SDK路径
        sdk_path = os.getenv('ANDROID_HOME')
        if not sdk_path:
            sdk_path = str(Path.home() / "Android/Sdk")

        # 获取NDK路径
        ndk_path = os.getenv('ANDROID_NDK_HOME')
        if not ndk_path:
            ndk_path = str(Path(sdk_path) / "ndk-bundle")

        # 写入local.properties
        properties_file = android_dir / "local.properties"
        with open(properties_file, 'w') as f:
            f.write(f"sdk.dir={sdk_path}\n")
            f.write(f"ndk.dir={ndk_path}\n")

    except Exception as e:
        raise Exception(f"Failed to update local.properties: {e}")

def build_android_project(android_dir: Path) -> bool:
    """构建Android项目"""
    try:
        # 确定gradlew路径
        gradlew = str(android_dir / ("gradlew.bat" if os.name == "nt" else "gradlew"))
        
        # 添加执行权限
        if os.name != "nt":
            os.chmod(gradlew, 0o755)

        # 执行构建
        process = subprocess.run(
            [gradlew, "assembleDebug"],
            cwd=str(android_dir),
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            logger.error(f"Build failed: {process.stderr}")
            return False

        return True

    except Exception as e:
        logger.error(f"Build process failed: {e}")
        return False