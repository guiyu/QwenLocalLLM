package com.example.qwentts;

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
}