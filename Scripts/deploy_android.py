# 文件路径: scripts/deploy_android.py

import logging
import shutil
from pathlib import Path
import subprocess
import os
import subprocess
import locale

logger = logging.getLogger(__name__)

def deploy_to_android(model_path: Path, android_dir: Path) -> bool:
    """部署GGML模型到Android项目"""
    try:
        # 1. 确保Android项目目录存在
        if not android_dir.exists():
            raise Exception(f"Android project directory not found: {android_dir}")

        # 2. 复制GGML模型文件到assets目录
        assets_dir = android_dir / "app/src/main/assets/models"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制并重命名为model.ggml
        target_path = assets_dir / "model.ggml"
        shutil.copy2(str(model_path), str(target_path))
        logger.info(f"Model copied to {target_path}")

        # 3. 确保GGML源码存在
        ggml_dir = android_dir / "app/src/main/cpp/ggml"
        if not ggml_dir.exists():
            os.makedirs(ggml_dir)
            # 这里应该复制GGML源码文件
            # TODO: 添加GGML源码复制逻辑

        # 4. 更新local.properties
        update_local_properties(android_dir)

        # 5. 执行Gradle构建
        if not build_android_project(android_dir):
            raise Exception("Android build failed")

        # 6. 验证APK是否生成
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
            ndk_path = str(Path(sdk_path) / "ndk/25.2.9519653")

        # 写入local.properties
        properties_file = android_dir / "local.properties"
        with open(properties_file, 'w') as f:
            f.write(f"sdk.dir={sdk_path}\n")
            f.write(f"ndk.dir={ndk_path}\n")
            f.write("cmake.dir=${sdk.dir}/cmake/3.22.1") # 添加CMake路径

    except Exception as e:
        raise Exception(f"Failed to update local.properties: {e}")

def build_android_project(android_dir: Path) -> bool:
    """构建 Android 项目"""
    try:
        # 确定 gradlew 路径
        gradlew = str(android_dir / ("gradlew.bat" if os.name == "nt" else "gradlew"))
        
        # 添加执行权限（非 Windows 系统）
        if os.name != "nt":
            os.chmod(gradlew, 0o755)
        
        # 执行 Gradle 构建
        process = subprocess.run(
            [gradlew, "clean", "assembleDebug"],
            cwd=str(android_dir),
            capture_output=True,
            text=True,  # 确保输出为文本格式
            encoding="utf-8"  # 强制使用 UTF-8 编码解析输出
        )
        
        if process.returncode != 0:
            logger.error(f"Build failed: {process.stderr}")
            return False
        
        logger.info("Build completed successfully.")
        return True
    
    except Exception as e:
        logger.error(f"Build process failed: {e}")
        return False