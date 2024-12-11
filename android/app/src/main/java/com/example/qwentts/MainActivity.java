package com.example.qwentts;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import ai.onnxruntime.OrtException;

public class MainActivity extends AppCompatActivity {
    private ModelLoader modelLoader;
    private EditText inputText;
    private Button runButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 初始化视图
        inputText = findViewById(R.id.input_text);
        runButton = findViewById(R.id.run_button);

        // 初始化模型加载器
        try {
            modelLoader = new ModelLoader(this);
        } catch (Exception e) {
            Toast.makeText(this, "Model initialization failed: " + e.getMessage(),
                         Toast.LENGTH_LONG).show();
            e.printStackTrace();
        }

        // 设置按钮点击事件
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runInference();
            }
        });
    }

    private void runInference() {
        String input = inputText.getText().toString();
        if (input.isEmpty()) {
            Toast.makeText(this, "Please enter some text",
                         Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            float[] result = modelLoader.runInference(input);
            // 处理结果...
            Toast.makeText(this, "Inference completed!",
                         Toast.LENGTH_SHORT).show();
        } catch (OrtException e) {
            Toast.makeText(this, "Inference failed: " + e.getMessage(),
                         Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (modelLoader != null) {
            modelLoader.close();
        }
    }
}