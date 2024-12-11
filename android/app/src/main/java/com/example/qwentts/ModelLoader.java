package com.example.qwentts;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ModelLoader {
    private static final String MODEL_FILE = "models/model_quantized.onnx";
    private final Context context;
    private OrtEnvironment env;
    private OrtSession session;

    public ModelLoader(Context context) throws IOException, OrtException {
        this.context = context;
        initModel();
    }

    private void initModel() throws IOException, OrtException {
        // 初始化ONNX Runtime环境
        env = OrtEnvironment.getEnvironment();
        
        // 创建会话选项
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        // 设置线程数
        sessionOptions.setIntraOpNumThreads(4);
        
        // 从assets加载模型
        byte[] modelBytes = loadModel();
        session = env.createSession(modelBytes, sessionOptions);
    }

    private byte[] loadModel() throws IOException {
        AssetManager assetManager = context.getAssets();
        InputStream inputStream = assetManager.open(MODEL_FILE);
        byte[] buffer = new byte[inputStream.available()];
        inputStream.read(buffer);
        inputStream.close();
        return buffer;
    }

    public float[] runInference(String input) throws OrtException {
        // 将输入转换为模型所需格式
        long[] inputShape = {1, input.length()};
        long[] inputIds = new long[input.length()];
        for (int i = 0; i < input.length(); i++) {
            inputIds[i] = input.charAt(i);
        }

        // 创建输入张量
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputIds, inputShape);
        
        // 准备输入Map
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputTensor);

        // 运行推理
        OrtSession.Result result = session.run(inputs);
        
        // 处理输出
        float[] outputData = ((float[][]) result.get(0).getValue())[0];
        
        // 清理资源
        inputTensor.close();
        result.close();
        
        return outputData;
    }

    public void close() {
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
}