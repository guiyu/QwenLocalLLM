#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include "ggml/ggml.h"

struct ModelContext {
    ggml_context* ctx = nullptr;
    std::vector<ggml_tensor*> tensors;
    size_t mem_size = 0;
    void* mem_buffer = nullptr;
};

// 全局模型上下文映射
static std::unordered_map<jlong, std::unique_ptr<ModelContext>> g_model_contexts;

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_qwentts_ModelLoader_initializeModel(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath) {
    
    const char* path = env->GetStringUTFChars(modelPath, 0);
    
    // 创建新的模型上下文
    auto ctx = std::make_unique<ModelContext>();
    
    // 初始化GGML
    ctx->mem_size = ggml_time_init(); // 获取所需内存大小
    ctx->mem_buffer = malloc(ctx->mem_size);
    
    if (!ctx->mem_buffer) {
        env->ReleaseStringUTFChars(modelPath, path);
        return 0;
    }
    
    // 创建GGML上下文
    ctx->ctx = ggml_init({
        .mem_size = ctx->mem_size,
        .mem_buffer = ctx->mem_buffer,
    });
    
    if (!ctx->ctx) {
        free(ctx->mem_buffer);
        env->ReleaseStringUTFChars(modelPath, path);
        return 0;
    }
    
    // 加载模型
    FILE* f = fopen(path, "rb");
    if (!f) {
        ggml_free(ctx->ctx);
        free(ctx->mem_buffer);
        env->ReleaseStringUTFChars(modelPath, path);
        return 0;
    }
    
    // 读取模型文件
    // TODO: 实现模型文件的具体读取逻辑
    
    fclose(f);
    env->ReleaseStringUTFChars(modelPath, path);
    
    // 存储上下文并返回指针
    jlong handle = reinterpret_cast<jlong>(ctx.get());
    g_model_contexts[handle] = std::move(ctx);
    
    return handle;
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_qwentts_ModelLoader_runInference(
        JNIEnv* env,
        jobject /* this */,
        jlong modelPtr,
        jstring input) {
    
    // 获取模型上下文
    auto it = g_model_contexts.find(modelPtr);
    if (it == g_model_contexts.end()) {
        return nullptr;
    }
    
    auto& ctx = it->second;
    const char* text = env->GetStringUTFChars(input, 0);
    
    // 准备输入数据
    std::vector<float> input_data;
    size_t text_len = strlen(text);
    input_data.resize(text_len);
    
    for (size_t i = 0; i < text_len; ++i) {
        input_data[i] = static_cast<float>(text[i]);
    }
    
    // 创建计算图
    ggml_cgraph gf = {};
    gf.n_threads = 4;  // 使用4个线程
    
    // TODO: 实现具体的推理计算逻辑
    
    // 创建返回数组
    jfloatArray result = env->NewFloatArray(input_data.size());
    env->SetFloatArrayRegion(result, 0, input_data.size(), input_data.data());
    
    env->ReleaseStringUTFChars(input, text);
    return result;
}

JNIEXPORT void JNICALL
Java_com_example_qwentts_ModelLoader_destroyModel(
        JNIEnv* env,
        jobject /* this */,
        jlong modelPtr) {
    
    // 清理模型资源
    auto it = g_model_contexts.find(modelPtr);
    if (it != g_model_contexts.end()) {
        auto& ctx = it->second;
        if (ctx->ctx) {
            ggml_free(ctx->ctx);
        }
        if (ctx->mem_buffer) {
            free(ctx->mem_buffer);
        }
        g_model_contexts.erase(it);
    }
}

} // extern "C"