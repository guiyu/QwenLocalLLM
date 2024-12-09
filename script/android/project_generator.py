import os
from pathlib import Path
import logging
import shutil
import json
import traceback

logger = logging.getLogger(__name__)

class AndroidProjectGenerator:
    def __init__(self, output_dir="./android"):
            self.output_dir = Path(output_dir)
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
            
            # 创建基本资源目录
            values_dir = app_dir / "src" / "main" / "res" / "values"
            mipmap_dir = app_dir / "src" / "main" / "res" / "mipmap-hdpi"
            
            # 创建所有必要的目录
            for dir_path in [java_dir, res_dir, assets_dir, values_dir, mipmap_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            
            # 生成基本资源文件
            strings_xml = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">QwenTTS</string>
</resources>"""
            (values_dir / "strings.xml").write_text(strings_xml, encoding='utf-8')
            logger.info("Generated strings.xml")
            
            # 生成简单图标文件
            icon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00\x00\x00\x02\x00\x01\x98\x98\xe2\xe6\x00\x00\x00\x00IEND\xaeB`\x82'
            (mipmap_dir / "ic_launcher.png").write_bytes(icon_data)
            (mipmap_dir / "ic_launcher_round.png").write_bytes(icon_data)
            logger.info("Generated launcher icons")
            
            paths = {
                "app_dir": str(app_dir),
                "java_dir": str(java_dir),
                "res_dir": str(res_dir),
                "assets_dir": str(assets_dir)
            }
            
            # 复制模板文件
            self.copy_templates(paths)
            
            # 生成构建文件
            self.generate_build_files(paths)
            
            logger.info("Project structure generated successfully")
            return paths
            
        except Exception as e:
            logger.error(f"Failed to generate project structure: {e}")
            raise

    def copy_templates(self, paths):
            """复制模板文件到项目目录"""
            try:
                logger.info("Copying template files...")

                activity_main_xml = """<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <EditText
        android:id="@+id/input_text"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:hint="Enter text to convert to speech"
        android:minHeight="100dp"
        android:gravity="top"
        android:inputType="textMultiLine" />

    <Button
        android:id="@+id/generate_button"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        android:text="Generate Speech" />

</LinearLayout>"""

        # 使用 UTF-8 编码写入文件
                res_dir = Path(paths["res_dir"])
                (res_dir / "activity_main.xml").write_text(activity_main_xml, encoding='utf-8')
                logger.info("Written activity_main.xml with UTF-8 encoding")
                
                # 获取模板目录
                templates_dir = Path(__file__).parent / "templates"
                logger.info(f"Template directory: {templates_dir}")
                
                # 复制Java文件
                # java_dir = Path(paths["java_dir"])
                # logger.info(f"Copying Java files to: {java_dir}")
                # shutil.copy2(templates_dir / "MainActivity.java", java_dir / "MainActivity.java")
                # shutil.copy2(templates_dir / "Utils.java", java_dir / "Utils.java")
                # logger.info("Copied Java template files")
                
                # 复制布局文件
                # res_dir = Path(paths["res_dir"])
                # logger.info(f"Copying layout files to: {res_dir}")
                # shutil.copy2(templates_dir / "activity_main.xml", res_dir / "activity_main.xml")
                # logger.info("Copied layout template files")

                # MainActivity.java 内容
                main_activity_content = """package com.example.qwentts;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import android.media.AudioTrack;
import android.media.AudioFormat;
import android.media.AudioManager;

public class MainActivity extends AppCompatActivity {
    private QwenTTSModel ttsModel;
    private EditText inputText;
    private Button generateButton;
    private AudioTrack audioTrack;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 初始化视图
        inputText = findViewById(R.id.input_text);
        generateButton = findViewById(R.id.generate_button);

        // 初始化模型
        try {
            ttsModel = new QwenTTSModel(this);
        } catch (Exception e) {
            Toast.makeText(this, "Error initializing model: " + e.getMessage(), 
                         Toast.LENGTH_LONG).show();
        }

        // 初始化音频播放器
        initAudioTrack();

        // 设置按钮点击事件
        generateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                generateAndPlaySpeech();
            }
        });
    }

    private void initAudioTrack() {
        int sampleRate = 22050;
        int minBufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        );

        audioTrack = new AudioTrack(
            AudioManager.STREAM_MUSIC,
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
            minBufferSize,
            AudioTrack.MODE_STREAM
        );
    }

    private void generateAndPlaySpeech() {
        String text = inputText.getText().toString();
        if (text.isEmpty()) {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            float[] audioData = ttsModel.generateSpeech(text);
            if (audioData != null) {
                audioTrack.play();
                audioTrack.write(audioData, 0, audioData.length, AudioTrack.WRITE_BLOCKING);
                audioTrack.stop();
            }
        } catch (Exception e) {
            Toast.makeText(this, "Error generating speech: " + e.getMessage(), 
                         Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (ttsModel != null) {
            ttsModel.close();
        }
        if (audioTrack != null) {
            audioTrack.release();
        }
    }
}"""

        # Utils.java 内容
                utils_content = """package com.example.qwentts;

import android.content.Context;
import java.io.InputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;

public class Utils {
    public static byte[] loadModelFromAssets(Context context, String modelPath) throws IOException {
        InputStream inputStream = context.getAssets().open(modelPath);
        byte[] buffer = new byte[inputStream.available()];
        inputStream.read(buffer);
        inputStream.close();
        return buffer;
    }

    public static float[] textToInputArray(String text) {
        float[] input = new float[text.length()];
        for (int i = 0; i < text.length(); i++) {
            input[i] = (float) text.charAt(i);
        }
        return input;
    }

    public static float[] processOutput(OrtSession.Result result) throws OrtException {
        Object outputTensor = result.get(0).getValue();
        if (outputTensor instanceof float[]) {
            return (float[]) outputTensor;
        }
        return new float[0];
    }
}"""

                # 写入文件
                java_dir = Path(paths["java_dir"])
                (java_dir / "MainActivity.java").write_text(main_activity_content, encoding='utf-8')
                (java_dir / "Utils.java").write_text(utils_content, encoding='utf-8')
                logger.info("Generated Java files")

                 # 添加 QwenTTSModel.java 内容
                qwen_tts_model_content = """package com.example.qwentts;

import android.content.Context;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.nio.FloatBuffer;

public class QwenTTSModel {
    private final OrtEnvironment env;
    private final OrtSession session;
    private static final String MODEL_FILE = "model_quantized.onnx";

    public QwenTTSModel(Context context) throws Exception {
        env = OrtEnvironment.getEnvironment();
        byte[] modelData = Utils.loadModelFromAssets(context, MODEL_FILE);
        session = env.createSession(modelData);
    }

    public float[] generateSpeech(String text) throws Exception {
        // 准备输入
        float[] inputData = Utils.textToInputArray(text);
        
        // 创建输入tensor
        long[] shape = new long[]{1, inputData.length};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);
        
        // 准备输入map
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputTensor);
        
        // 运行推理
        OrtSession.Result result = session.run(inputs);
        
        // 处理输出
        float[] output = Utils.processOutput(result);
        
        // 释放资源
        inputTensor.close();
        
        return output;
    }

    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if (env != null) {
            try {
                env.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}"""

                # 写入文件
                java_dir = Path(paths["java_dir"])
                (java_dir / "QwenTTSModel.java").write_text(qwen_tts_model_content, encoding='utf-8')
                logger.info("Generated QwenTTSModel.java")
    
                return True 
                

            except Exception as e:
                logger.error(f"Failed to copy template files: {e}")
                raise
    
    def generate_build_files(self, paths):
        """生成构建配置文件"""
        try:
            logger.info("Generating build files...")
            
            # 生成 settings.gradle
            settings_gradle = """
rootProject.name = 'qwentts'
include ':app'
"""
            (self.output_dir / "settings.gradle").write_text(settings_gradle.strip())
            logger.info("Generated settings.gradle")

            # 生成根目录 build.gradle
            root_build_gradle = """
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:8.1.0'
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://maven.aliyun.com/repository/google' }
        maven { url 'https://maven.aliyun.com/repository/public' }
    }
}
"""
            (self.output_dir / "build.gradle").write_text(root_build_gradle.strip())
            logger.info("Generated root build.gradle")

            # 生成 app/build.gradle
            app_build_gradle = """
plugins {
    id 'com.android.application'
}

android {
    namespace 'com.example.qwentts'
    compileSdk 33
    
    defaultConfig {
        applicationId "com.example.qwentts"
        minSdk 24
        targetSdk 33
        versionCode 1
        versionName "1.0"
        
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
}

dependencies {
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.9.0'
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.14.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
}
"""
            (Path(paths["app_dir"]) / "build.gradle").write_text(app_build_gradle.strip())
            logger.info("Generated app build.gradle")

            # 添加生成 gradle.properties 文件
            gradle_properties = """
android.useAndroidX=true
android.enableJetifier=true
org.gradle.jvmargs=-Xmx8g -Dfile.encoding=UTF-8
org.gradle.parallel=true
org.gradle.daemon=true
org.gradle.caching=true
android.enableR8.fullMode=false
android.defaults.buildfeatures.buildconfig=true
"""
            (self.output_dir / "gradle.properties").write_text(gradle_properties.strip())
            logger.info("Generated gradle.properties")
            
            # 生成 gradle wrapper
            self._generate_gradle_wrapper()
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate build files: {e}")
            raise

    def _generate_gradle_wrapper(self):
        """生成 Gradle Wrapper 文件"""
        try:
            logger.info("Generating Gradle Wrapper...")
            
            # 创建 gradle/wrapper 目录
            wrapper_dir = self.output_dir / "gradle" / "wrapper"
            wrapper_dir.mkdir(parents=True, exist_ok=True)
            
            # gradle-wrapper.properties
            wrapper_properties = """
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\\://mirrors.cloud.tencent.com/gradle/gradle-8.11-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
"""
            (wrapper_dir / "gradle-wrapper.properties").write_text(wrapper_properties.strip())
            logger.info("Generated gradle-wrapper.properties")

            # 复制 gradle-wrapper.jar（这个文件需要从有效的Android项目中复制）
            # TODO: 添加gradle-wrapper.jar的复制逻辑

            logger.info("Gradle Wrapper generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate Gradle Wrapper: {e}")
            raise
    
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
        generator.generate_build_files(paths)
        logger.info("构建文件生成完成")

        # 4. 生成AndroidManifest.xml
        generator.generate_manifest()
        logger.info("AndroidManifest.xml生成完成")

        # 5. 生成gradle wrapper
        generator._generate_gradle_wrapper()
        logger.info("Gradle Wrapper生成完成")

        
        return paths
        
    except Exception as e:
        logger.error(f"Android项目生成失败: {e}")
        raise