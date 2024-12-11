# 文件路径: scripts/build_ggml.py

import logging
import subprocess
import os
from pathlib import Path
import shutil
import sys

logger = logging.getLogger(__name__)

def build_ggml(project_root: Path) -> bool:
    """编译GGML库和工具"""
    try:
        # 1. 克隆GGML仓库
        ggml_dir = project_root / "third_party/ggml"
        if not ggml_dir.exists():
            logger.info("Cloning GGML repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/ggml.git", str(ggml_dir)],
                check=True
            )

        # 2. 创建构建目录
        build_dir = ggml_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # 3. 运行CMake配置
        logger.info("Configuring CMake...")
        cmake_cmd = [
            "cmake",
            "-B", str(build_dir),
            "-S", str(ggml_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_BUILD_EXAMPLES=ON",
            "-DGGML_BUILD_TESTS=OFF"
        ]
        
        subprocess.run(cmake_cmd, check=True)

        # 4. 编译
        logger.info("Building GGML...")
        build_cmd = [
            "cmake",
            "--build", str(build_dir),
            "--config", "Release"
        ]
        subprocess.run(build_cmd, check=True)

        # 5. 复制必要文件到项目目录
        logger.info("Copying files...")
        # 复制量化工具
        quantize_exe = build_dir / ("quantize.exe" if sys.platform == "win32" else "quantize")
        target_dir = project_root / "ggml"
        target_dir.mkdir(exist_ok=True)
        
        if quantize_exe.exists():
            shutil.copy2(str(quantize_exe), str(target_dir))
        else:
            raise Exception("Quantize tool not found")

        # 复制头文件和源文件到Android项目
        android_cpp_dir = project_root / "android/app/src/main/cpp/ggml"
        android_cpp_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制必要的源文件
        for file in ["ggml.h", "ggml.c"]:
            shutil.copy2(
                str(ggml_dir / file),
                str(android_cpp_dir / file)
            )

        logger.info("GGML build completed successfully")
        return True

    except Exception as e:
        logger.error(f"GGML build failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).parent.parent
    build_ggml(project_root)