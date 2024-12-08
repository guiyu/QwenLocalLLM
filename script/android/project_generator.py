# 文件路径: script/android/project_generator.py
# 新建文件

import os
from pathlib import Path
import logging
import shutil
import json
import traceback

logger = logging.getLogger(__name__)

class AndroidProjectGenerator:
    def __init__(self, output_dir="./android"):
            self.output_dir = Path(output_dir).resolve()
            logger.info(f"Initializing Android project generator at: {self.output_dir}")
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created output directory successfully")
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
                raise
    
    def generate_project_structure(self):
            """生成Android项目基本结构"""
            try:
                logger.info("Generating Android project structure...")
                
                # 创建目录结构
                app_dir = self.output_dir / "app"
                java_dir = app_dir / "src" / "main" / "java" / "com" / "example" / "qwentts"
                res_dir = app_dir / "src" / "main" / "res" / "layout"
                assets_dir = app_dir / "src" / "main" / "assets"
                
                # 创建所有必要的目录
                for dir_path in [java_dir, res_dir, assets_dir]:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                
                paths = {
                    "app_dir": str(app_dir),
                    "java_dir": str(java_dir),
                    "res_dir": str(res_dir),
                    "assets_dir": str(assets_dir)
                }
                
                # 添加这行：复制模板文件
                self.copy_templates(paths)
                
                logger.info("Project structure generated successfully")
                return paths
                
            except Exception as e:
                logger.error(f"Failed to generate project structure: {e}")
                raise

    def copy_templates(self, paths):
        """复制模板文件到项目目录"""
        try:
            logger.info("Copying template files...")
            
            # 获取模板目录
            templates_dir = Path(__file__).parent / "templates"
            logger.info(f"Template directory: {templates_dir}")
            
            if not templates_dir.exists():
                raise FileNotFoundError(f"Templates directory not found at {templates_dir}")
            
            # 复制Java文件
            java_dir = Path(paths["java_dir"])
            logger.info(f"Copying Java files to: {java_dir}")
            
            # 确保目标目录存在
            java_dir.mkdir(parents=True, exist_ok=True)
            
            # 详细的文件复制操作
            for java_file in ["MainActivity.java", "Utils.java"]:
                src = templates_dir / java_file
                dst = java_dir / java_file
                if not src.exists():
                    raise FileNotFoundError(f"Template file not found: {src}")
                shutil.copy2(src, dst)
                logger.info(f"Copied {java_file}")
            
            # 复制布局文件
            res_dir = Path(paths["res_dir"])
            logger.info(f"Copying layout files to: {res_dir}")
            
            # 确保目标目录存在
            res_dir.mkdir(parents=True, exist_ok=True)
            
            layout_file = "activity_main.xml"
            src = templates_dir / layout_file
            dst = res_dir / layout_file
            if not src.exists():
                raise FileNotFoundError(f"Template file not found: {src}")
            shutil.copy2(src, dst)
            logger.info(f"Copied {layout_file}")
            
            logger.info("All template files copied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy template files: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate_build_files(self):
        """生成构建配置文件"""
        # 生成根目录build.gradle
        root_build_gradle = """
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:7.0.4'
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}
"""
        
        # 生成app/build.gradle
        app_build_gradle = """
plugins {
    id 'com.android.application'
}

android {
    compileSdkVersion 31
    
    defaultConfig {
        applicationId "com.example.qwentts"
        minSdkVersion 21
        targetSdkVersion 31
        versionCode 1
        versionName "1.0"
    }
    
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}

dependencies {
    implementation 'androidx.appcompat:appcompat:1.4.1'
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.14.0'
}
"""
        
        # 写入构建文件
        with open(self.output_dir / "build.gradle", "w") as f:
            f.write(root_build_gradle)
        
        with open(self.output_dir / "app" / "build.gradle", "w") as f:
            f.write(app_build_gradle)
        
        return True
    
    def generate_manifest(self):
        """生成AndroidManifest.xml"""
        manifest = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.qwentts">

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.AppCompat.Light">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
"""
        
        manifest_path = self.output_dir / "app" / "src" / "main" / "AndroidManifest.xml"
        with open(manifest_path, "w") as f:
            f.write(manifest)
        
        return str(manifest_path)

def generate_android_project(config):
    try:
        logger.info("开始生成Android项目结构...")
        
        generator = AndroidProjectGenerator(config.ANDROID_OUTPUT_DIR)
        logger.info(f"项目输出目录: {config.ANDROID_OUTPUT_DIR}")
        
        # 1. 生成项目结构
        paths = generator.generate_project_structure()
        logger.info("项目结构生成完成")
        
        # 2. 复制模板文件
        generator.copy_templates(paths)
        logger.info("模板文件复制完成")
        
        # 3. 生成构建文件
        generator.generate_build_files()
        logger.info("构建文件生成完成")
        
        return paths
        
    except Exception as e:
        logger.error(f"Android项目生成失败: {e}")
        raise