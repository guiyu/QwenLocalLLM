package com.example.qwentts;

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
}