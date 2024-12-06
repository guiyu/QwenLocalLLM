# 文件路径: script/android/templates/MainActivity.java
# 新建文件

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

    private void generateAndPlaySpeech() {
        String text = inputText.getText().toString();
        if (text.isEmpty()) {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // 生成语音
            float[] audioData = ttsModel.generateSpeech(text);
            if (audioData != null) {
                // 播放音频
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
}