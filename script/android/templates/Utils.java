# 文件路径: script/android/templates/Utils.java
# 新建文件

package com.example.qwentts;

import android.content.Context;
import java.io.InputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import ai.onnxruntime.OrtSession;

public class Utils {
    public static byte[] loadModelFromAssets(Context context, String modelPath) throws IOException {
        InputStream inputStream = context.getAssets().open(modelPath);
        byte[] buffer = new byte[inputStream.available()];
        inputStream.read(buffer);
        inputStream.close();
        return buffer;
    }

    public static float[] textToInputArray(String text) {
        // 简单的文本转数字数组实现
        float[] input = new float[text.length()];
        for (int i = 0; i < text.length(); i++) {
            input[i] = (float) text.charAt(i);
        }
        return input;
    }

    public static float[] processOutput(OrtSession.Result result) {
        // 处理模型输出为音频数据
        Object outputTensor = result.get(0).getValue();
        if (outputTensor instanceof float[]) {
            return (float[]) outputTensor;
        }
        return new float[0];
    }
}