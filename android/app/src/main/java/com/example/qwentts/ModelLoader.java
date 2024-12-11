package com.example.qwentts;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ModelLoader {
    private static final String TAG = "ModelLoader";
    private static final String MODEL_FILE = "models/model.ggml";
    
    // 加载native库
    static {
        System.loadLibrary("ggml");
        System.loadLibrary("phi2");
    }

    private final Context context;
    private long modelPtr;  // 本地模型指针

    public ModelLoader(Context context) throws IOException {
        this.context = context;
        initModel();
    }

    private native long initializeModel(String modelPath);
    private native float[] runInference(long modelPtr, String input);
    private native void destroyModel(long modelPtr);

    private void initModel() throws IOException {
        // 从assets复制模型到本地存储
        File modelFile = new File(context.getFilesDir(), MODEL_FILE);
        if (!modelFile.exists()) {
            copyModelFromAssets();
        }

        // 初始化模型
        modelPtr = initializeModel(modelFile.getAbsolutePath());
        if (modelPtr == 0) {
            throw new IOException("Failed to initialize model");
        }
    }

    private void copyModelFromAssets() throws IOException {
        File modelFile = new File(context.getFilesDir(), MODEL_FILE);
        modelFile.getParentFile().mkdirs();

        AssetManager assetManager = context.getAssets();
        try (InputStream in = assetManager.open(MODEL_FILE);
             FileOutputStream out = new FileOutputStream(modelFile)) {
            
            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
    }

    public float[] runModelInference(String input) {
        if (modelPtr == 0) {
            throw new IllegalStateException("Model not initialized");
        }
        return runInference(modelPtr, input);
    }

    public void close() {
        if (modelPtr != 0) {
            destroyModel(modelPtr);
            modelPtr = 0;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }
}