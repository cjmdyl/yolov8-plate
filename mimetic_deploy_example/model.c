#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define NUM_CLASSES 10

// 模型结构
typedef struct {
    float* W1;  // [HIDDEN_SIZE * INPUT_SIZE]
    float* b1;  // [HIDDEN_SIZE]
    float* W2;  // [NUM_CLASSES * HIDDEN_SIZE]
    float* b2;  // [NUM_CLASSES]
} Model;

// 加载训练好的模型权重
Model* load_model(const char* weights_file) {
    FILE* file = fopen(weights_file, "r");
    if (!file) {
        printf("Error: Cannot open weights file %s\n", weights_file);
        return NULL;
    }

    Model* model = (Model*)malloc(sizeof(Model));
    model->W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    model->b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    model->W2 = (float*)malloc(NUM_CLASSES * HIDDEN_SIZE * sizeof(float));
    model->b2 = (float*)malloc(NUM_CLASSES * sizeof(float));

    char line[256];
    float value;
    int index = 0;

    // 读取权重
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "fc1.weight")) {
            for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
                fscanf(file, "%f", &value);
                model->W1[i] = value;
            }
        }
        else if (strstr(line, "fc1.bias")) {
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                fscanf(file, "%f", &value);
                model->b1[i] = value;
            }
        }
        else if (strstr(line, "fc2.weight")) {
            for (int i = 0; i < NUM_CLASSES * HIDDEN_SIZE; i++) {
                fscanf(file, "%f", &value);
                model->W2[i] = value;
            }
        }
        else if (strstr(line, "fc2.bias")) {
            for (int i = 0; i < NUM_CLASSES; i++) {
                fscanf(file, "%f", &value);
                model->b2[i] = value;
            }
        }
    }

    fclose(file);
    printf("Model loaded successfully!\n");
    return model;
}

// 释放模型内存
void free_model(Model* model) {
    if (model) {
        free(model->W1);
        free(model->b1);
        free(model->W2);
        free(model->b2);
        free(model);
    }
}

// 图像预处理
float* preprocess_image(const unsigned char* image_data, int width, int height) {
    float* processed = (float*)malloc(INPUT_SIZE * sizeof(float));
    
    for (int i = 0; i < INPUT_SIZE; i++) {
        // 归一化到 [0, 1]
        processed[i] = image_data[i] / 255.0f;
    }
    
    return processed;
}

// 打印预测结果
void print_prediction(const float* probabilities) {
    printf("Prediction probabilities:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("  %d: %.2f%%\n", i, probabilities[i] * 100);
    }
}

void matrix_vector_multiply(float* result, const float* matrix, 
                           const float* vector, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

void add(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void relu(float* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = vector[i] > 0 ? vector[i] : 0.0f;
    }
}

void softmax(float* array, int size) {
    float max_val = array[0];
    float sum = 0.0f;
    
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) max_val = array[i];
    }
    
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i] - max_val);
        sum += array[i];
    }
    
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}

int argmax(const float* array, int size) {
    int max_index = 0;
    float max_value = array[0];
    
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }
    
    return max_index;
}

// 推理
int forward(Model* model, const float* input) {
    float hidden[HIDDEN_SIZE];
    float output[NUM_CLASSES];
    
    // 第一层: W1 * input + b1
    mimeGemm8(/*A B C relu D*/);
    
    // 第二层: W2 * hidden + b2
    mimeGemm8(/*A B C sofrmax D*/);
    
    // 打印预测概率
    print_prediction(output);
    
    return argmax(output, NUM_CLASSES);
}

unsigned char* load_image_from_file(const char* filename, int* width, int* height) {
    // 这里应该是实际的图像加载代码
    // 返回一个包含图像数据的数组
    static unsigned char dummy_image[INPUT_SIZE];
    
    // 创建一些测试数据（实际中应该从文件读取）
    for (int i = 0; i < INPUT_SIZE; i++) {
        dummy_image[i] = rand() % 256; // 随机像素值
    }
    
    *width = 28;
    *height = 28;
    return dummy_image;
}

int main() {
    printf("MNIST手写数字识别推理部署\n");
    
    // 加载模型
    Model* model = load_model("model_weights.txt");
    if (!model) {
        return 1;
    }
    
    // 加载测试图像
    int width, height;
    unsigned char* image_data = load_image_from_file("test_image.bin", &width, &height);
    
    if (width != 28 || height != 28) {
        printf("Error: Image must be 28x28 pixels\n");
        free_model(model);
        return 1;
    }
    
    // 预处理图像
    float* processed_image = preprocess_image(image_data, width, height);
    
    // 进行推理
    int prediction = forward(model, processed_image);
    
    printf("\n=== 推理结果 ===\n");
    printf("预测数字: %d\n", prediction);
    
    // 清理内存
    free(processed_image);
    free_model(model);
    
    return 0;
}

