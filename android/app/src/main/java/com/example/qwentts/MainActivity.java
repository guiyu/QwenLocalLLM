package com.example.qwentts;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import android.media.AudioTrack;
import android.media.AudioFormat;
import android.media.AudioManager;
import com.example.qwentts.inference.ModelInference;

import java.util.concurrent.Future;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private EditText inputText;
    private Button runButton;
    private AudioTrack audioTrack;
    private ModelInference modelInference;

    private static final int PERMISSION_REQUEST_CODE = 1001;

    private boolean checkAndRequestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ArrayList<String> permissions = new ArrayList<>();
            
            // 检查所需权限
            if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) 
                != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.RECORD_AUDIO);
            }
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) 
                != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            }
            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) 
                != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            }
            
            if (!permissions.isEmpty()) {
                requestPermissions(permissions.toArray(new String[0]), 
                                PERMISSION_REQUEST_CODE);
                return false;
            }
        }
        return true;
    }

    private ProgressDialog loadingDialog;

    private void showLoadingDialog(String message) {
        if (loadingDialog == null) {
            loadingDialog = new ProgressDialog(this);
            loadingDialog.setCancelable(false);
            loadingDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
        }
        loadingDialog.setMessage(message);
        loadingDialog.show();
    }

    private void updateLoadingProgress(String message) {
        runOnUiThread(() -> {
            if (loadingDialog != null && loadingDialog.isShowing()) {
                loadingDialog.setMessage(message);
            }
        });
    }

    private void hideLoadingDialog() {
        runOnUiThread(() -> {
            if (loadingDialog != null && loadingDialog.isShowing()) {
                loadingDialog.dismiss();
            }
        });
    }

    private void initializeModel() {
        showLoadingDialog("Initializing model...");
        new Thread(() -> {
            try {
                updateLoadingProgress("Loading model files...");
                modelInference = new ModelInference(this);
                
                updateLoadingProgress("Preparing inference engine...");
                // 初始化推理引擎
                
                hideLoadingDialog();
                runOnUiThread(() -> {
                    Toast.makeText(this, "Model loaded successfully", 
                                Toast.LENGTH_SHORT).show();
                    enableUI();
                });
            } catch (Exception e) {
                hideLoadingDialog();
                runOnUiThread(() -> {
                    Toast.makeText(this, "Model initialization failed: " + e.getMessage(),
                                Toast.LENGTH_LONG).show();
                    disableUI();
                });
            }
        }).start();
    }

    private void enableUI() {
        runButton.setEnabled(true);
        voiceInputButton.setEnabled(true);
    }

    private void disableUI() {
        runButton.setEnabled(false);
        voiceInputButton.setEnabled(false);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                        @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            
            if (allGranted) {
                initializeModel();
            } else {
                Toast.makeText(this, "Required permissions not granted", 
                            Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
        
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 添加语音输入按钮
        voiceInputButton = findViewById(R.id.voice_input_button);
        voiceInputButton.setOnClickListener(v -> toggleRecording());
        
        // 请求权限
        requestPermissions(new String[]{
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        }, 1);
        
        // 初始化视图
        inputText = findViewById(R.id.input_text);
        runButton = findViewById(R.id.run_button);
        
        // 初始化模型推理
        try {
            modelInference = new ModelInference(this);
        } catch (Exception e) {
            Toast.makeText(this, "Model initialization failed: " + e.getMessage(), 
                         Toast.LENGTH_LONG).show();
            e.printStackTrace();
        }
        
        // 初始化音频播放器
        initAudioTrack();
        
        // 设置按钮点击事件
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runInference();
            }
        });
    }
    
    private void initAudioTrack() {
        int sampleRate = 22050; // 采样率
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
    
    private void runInference() {
        String text = inputText.getText().toString();
        if (text.isEmpty()) {
            Toast.makeText(this, "Please enter some text", 
                         Toast.LENGTH_SHORT).show();
            return;
        }
        
        // 禁用按钮
        runButton.setEnabled(false);
        
        try {
            // 异步运行推理
            Future<float[]> future = modelInference.runInference(text);
            
            // 在后台线程中等待结果
            new Thread(() -> {
                try {
                    // 获取推理结果
                    float[] audioData = future.get();
                    
                    // 在主线程中更新UI和播放音频
                    runOnUiThread(() -> {
                        playAudio(audioData);
                        runButton.setEnabled(true);
                    });
                    
                } catch (Exception e) {
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this,
                            "Inference failed: " + e.getMessage(),
                            Toast.LENGTH_SHORT).show();
                        runButton.setEnabled(true);
                    });
                }
            }).start();
            
        } catch (Exception e) {
            Toast.makeText(this, "Error running inference: " + e.getMessage(),
                         Toast.LENGTH_SHORT).show();
            runButton.setEnabled(true);
        }
    }
    
    private void playAudio(float[] audioData) {
        try {
            // 开始播放
            audioTrack.play();
            
            // 写入音频数据
            audioTrack.write(audioData, 0, audioData.length,
                           AudioTrack.WRITE_BLOCKING);
            
            // 等待播放完成
            audioTrack.stop();
            
        } catch (Exception e) {
            Toast.makeText(this, "Audio playback failed: " + e.getMessage(),
                         Toast.LENGTH_SHORT).show();
        }
    }

    private void toggleRecording() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
        isRecording = !isRecording;
    }

    private void startRecording() {
        // 开始录音逻辑
    }
    
    private void stopRecording() {
        // 停止录音并处理音频
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (modelInference != null) {
            modelInference.release();
        }
        if (audioTrack != null) {
            audioTrack.release();
        }
    }
}