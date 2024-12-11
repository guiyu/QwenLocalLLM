// 文件路径: android/app/src/main/java/com/example/qwentts/inference/ModelInference.java

package com.example.qwentts.inference;

import android.content.Context;
import android.util.Log;
import ai.onnxruntime.*;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.*;

public class ModelInference {
    private static final String TAG = "ModelInference";
    
    private final Context context;
    private OrtEnvironment env;
    private OrtSession session;
    private final ExecutorService executorService;
    
    // KV缓存管理
    private final Map<Integer, float[]> kvCache;
    private final int maxCacheSize = 2048;
    
    public ModelInference(Context context) {
        this.context = context;
        this.kvCache = new LRUMap<>(maxCacheSize);
        this.executorService = Executors.newFixedThreadPool(
            Runtime.getRuntime().availableProcessors()
        );
        initializeModel();
    }
    
    private void initializeModel() {
        try {
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            
            // 配置内存和线程
            sessionOptions.setIntraOpNumThreads(4);
            sessionOptions.setInterOpNumThreads(2);
            sessionOptions.setMemoryPatternOptimization(true);
            sessionOptions.setExecutionMode(ExecutionMode.SEQUENTIAL);
            
            // 加载模型
            String modelPath = Utils.copyModelToCache(context);
            session = env.createSession(modelPath, sessionOptions);
            
        } catch (Exception e) {
            Log.e(TAG, "Model initialization failed", e);
        }
    }
    
    public Future<float[]> runInference(String text) {
        return executorService.submit(() -> {
            try {
                // 1. 准备输入
                OnnxTensor inputTensor = prepareInput(text);
                
                                // 2. 运行推理
                OrtSession.Result result = session.run(
                    Collections.singletonMap("input_ids", inputTensor)
                );
                
                // 3. 处理输出
                float[] output = processOutput(result);
                
                // 4. 更新KV缓存
                updateKVCache(output);
                
                // 5. 清理资源
                inputTensor.close();
                result.close();
                
                return output;
                
            } catch (Exception e) {
                Log.e(TAG, "Inference failed", e);
                throw new RuntimeException("Inference failed", e);
            }
        });
    }
    
    private OnnxTensor prepareInput(String text) throws OrtException {
        try {
            // 将文本转换为输入IDs
            long[] inputIds = new long[text.length()];
            for (int i = 0; i < text.length(); i++) {
                inputIds[i] = text.charAt(i);
            }
            
            // 准备输入形状
            long[] shape = {1, inputIds.length};
            
            // 创建输入张量
            return OnnxTensor.createTensor(env, inputIds, shape);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to prepare input", e);
            throw new OrtException("Input preparation failed");
        }
    }
    
    private float[] processOutput(OrtSession.Result result) throws OrtException {
        try {
            // 获取输出张量
            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            
            // 转换为float数组
            float[][] outputArray = (float[][]) outputTensor.getValue();
            return outputArray[0]; // 取第一个batch的结果
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to process output", e);
            throw new OrtException("Output processing failed");
        }
    }
    
    private void updateKVCache(float[] output) {
        try {
            // 生成缓存键
            int cacheKey = Arrays.hashCode(output);
            
            // 更新缓存
            synchronized (kvCache) {
                kvCache.put(cacheKey, output);
            }
            
            // 如果缓存过大,清理旧的条目
            if (kvCache.size() > maxCacheSize) {
                pruneCache();
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to update KV cache", e);
        }
    }
    
    private void pruneCache() {
        synchronized (kvCache) {
            // 如果是LRUMap,会自动移除最旧的条目
            // 这里可以添加额外的清理逻辑
            if (kvCache.size() > maxCacheSize * 0.8) { // 80%阈值
                // 触发GC
                System.gc();
            }
        }
    }
    
    public void release() {
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
            executorService.shutdown();
        } catch (Exception e) {
            Log.e(TAG, "Failed to release resources", e);
        }
    }
    
    // 内部工具类
    private static class Utils {
        public static String copyModelToCache(Context context) {
            try {
                File cacheDir = new File(context.getCacheDir(), "models");
                cacheDir.mkdirs();
                
                File modelFile = new File(cacheDir, "model_quantized.onnx");
                if (!modelFile.exists()) {
                    // 从assets复制模型
                    java.io.InputStream in = context.getAssets()
                        .open("models/model_quantized.onnx");
                    java.io.FileOutputStream out = new java.io.FileOutputStream(modelFile);
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                    in.close();
                    out.close();
                }
                return modelFile.getAbsolutePath();
                
            } catch (Exception e) {
                Log.e(TAG, "Failed to copy model file", e);
                throw new RuntimeException("Model file copy failed", e);
            }
        }
    }
    
    // LRU缓存实现
    private static class LRUMap<K,V> extends LinkedHashMap<K,V> {
        private final int maxSize;
        
        public LRUMap(int maxSize) {
            super(16, 0.75f, true);
            this.maxSize = maxSize;
        }
        
        @Override
        protected boolean removeEldestEntry(Map.Entry<K,V> eldest) {
            return size() > maxSize;
        }
    }
}