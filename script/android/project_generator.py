# 文件路径: script/android/project_generator.py
# 新建文件

import os
from pathlib import Path
import logging
import shutil
import json

logger = logging.getLogger(__name__)

class AndroidProjectGenerator:
    def __init__(self, output_dir="./android"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_project_structure(self):
        """生成Android项目基本结构"""
        # 创建目录结构
        app_dir = self.output_dir / "app"
        java_dir = app_dir / "src" / "main" / "java" / "com" / "example" / "qwentts"
        res_dir = app_dir / "src" / "main" / "res"
        assets_dir = app_dir / "src" / "main" / "assets"
        
        for dir_path in [java_dir, res_dir, assets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "app_dir": str(app_dir),
            "java_dir": str(java_dir),
            "res_dir": str(res_dir),
            "assets_dir": str(assets_dir)
        }
    
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
        logger.info("Generating Android project structure...")
        
        generator = AndroidProjectGenerator(config.ANDROID_OUTPUT_DIR)
        
        # 1. 生成项目结构
        paths = generator.generate_project_structure()
        
        # 2. 生成构建文件
        generator.generate_build_files()
        
        # 3. 生成清单文件
        manifest_path = generator.generate_manifest()
        
        logger.info("Android project generated successfully!")
        logger.info(f"Project directory: {generator.output_dir}")
        
        return paths
        
    except Exception as e:
        logger.error(f"Error generating Android project: {str(e)}")
        raise

def copy_templates(self):
    """复制模板文件到Android项目"""
    templates_dir = Path(__file__).parent / "templates"
    
    # 复制Java文件
    shutil.copy2(
        templates_dir / "MainActivity.java",
        self.java_dir / "MainActivity.java"
    )
    shutil.copy2(
        templates_dir / "Utils.java",
        self.java_dir / "Utils.java"
    )
    
    # 复制布局文件
    layout_dir = self.res_dir / "layout"
    layout_dir.mkdir(exist_ok=True)
    shutil.copy2(
        templates_dir / "activity_main.xml",
        layout_dir / "activity_main.xml"
    )