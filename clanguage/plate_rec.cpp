#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 宏定义和全局变量
// 车牌字符集，包含了省份简称、数字、字母、特殊字符等
const char* plate_chars = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品";
// 预设的车牌颜色列表
const char* color_list[] = {"黑色", "蓝色", "绿色", "白色", "黄色"};

// 图像预处理的归一化参数，用于将像素值标准化到0均值和1标准差附近
const float mean_value = 0.588f;
const float std_value = 0.193f;

// ---
// BMP文件头和信息头结构体
// 使用 #pragma pack(push, 1) 确保结构体按1字节对齐，防止因编译器对齐而产生填充字节
#pragma pack(push, 1)
typedef struct {
    unsigned short type;        // 文件类型，通常为BM（0x4D42）
    unsigned int size;          // 文件总大小
    unsigned short reserved1;   // 保留字
    unsigned short reserved2;   // 保留字
    unsigned int offset;        // 像素数据偏移量
} BMPFileHeader;

typedef struct {
    unsigned int size;              // 信息头大小
    int width;                      // 图像宽度
    int height;                     // 图像高度
    unsigned short planes;          // 平面数
    unsigned short bit_count;       // 每像素位数
    unsigned int compression;       // 压缩方式
    unsigned int image_size;        // 图像数据大小
    int x_pixels_per_meter;       // 水平分辨率
    int y_pixels_per_meter;       // 垂直分辨率
    unsigned int colors_used;       // 使用的颜色数
    unsigned int colors_important;  // 重要颜色数
} BMPInfoHeader;
#pragma pack(pop)

// ---
// 自定义矩阵结构体，用于存储图像和模型数据
typedef struct {
    int rows;       // 行数
    int cols;       // 列数
    int channels;   // 通道数（如RGB的3通道）
    float* data;    // 指向实际数据的指针
} Matrix;

// 创建矩阵并分配内存
Matrix create_matrix(int rows, int cols, int channels) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.channels = channels;
    // 分配一块连续的内存空间
    mat.data = (float*)malloc(rows * cols * channels * sizeof(float));
    if (mat.data == NULL) {
        fprintf(stderr, "内存分配失败\n");
        exit(1);
    }
    return mat;
}

// 释放矩阵内存
void free_matrix(Matrix* mat) {
    if (mat->data) {
        free(mat->data);
        mat->data = NULL; // 防止悬空指针
    }
}

// ---
// 图像处理函数
// 读取BMP图像文件并转换为Matrix结构体
Matrix read_bmp(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "无法打开文件: %s\n", path);
        exit(1);
    }
    
    BMPFileHeader file_header;
    BMPInfoHeader info_header;
    
    // 读取文件头和信息头
    fread(&file_header, sizeof(BMPFileHeader), 1, file);
    fread(&info_header, sizeof(BMPInfoHeader), 1, file);
    
    // 检查文件类型和位深，只支持24位BMP
    if (file_header.type != 0x4D42) {
        fprintf(stderr, "不是有效的BMP文件: %s\n", path);
        fclose(file);
        exit(1);
    }
    
    if (info_header.bit_count != 24) {
        fprintf(stderr, "只支持24位BMP文件: %s\n", path);
        fclose(file);
        exit(1);
    }
    
    int width = info_header.width;
    int height = abs(info_header.height); // BMP高度可能是负数（从上到下存储）
    int channels = 3;
    
    // 计算BMP文件每行实际占用的字节数，必须是4的倍数
    int row_size = (width * channels + 3) & ~3;
    
    // 分配内存并读取原始像素数据
    unsigned char* bmp_data = (unsigned char*)malloc(row_size * height);
    fseek(file, file_header.offset, SEEK_SET);
    fread(bmp_data, 1, row_size * height, file);
    fclose(file);
    
    // 创建目标矩阵并进行数据转换
    Matrix img = create_matrix(height, width, channels);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // BMP数据是自下而上存储，这里进行y轴翻转
            int src_idx = (height - 1 - y) * row_size + x * channels;
            int dst_idx = (y * width + x) * channels;
            
            // BMP存储顺序是BGR，转换为RGB并归一化到0-1范围
            img.data[dst_idx + 2] = bmp_data[src_idx] / 255.0f;     // B -> R
            img.data[dst_idx + 1] = bmp_data[src_idx + 1] / 255.0f; // G -> G
            img.data[dst_idx] = bmp_data[src_idx + 2] / 255.0f;     // R -> B
        }
    }
    
    free(bmp_data);
    return img;
}

// 保存Matrix结构体为BMP图像文件
void save_bmp(const char* path, Matrix img) {
    int row_size = (img.cols * 3 + 3) & ~3;
    int image_size = row_size * img.rows;
    
    // 填充BMP文件头和信息头
    BMPFileHeader file_header = {
        .type = 0x4D42,
        .size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size,
        .offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader)
    };
    
    BMPInfoHeader info_header = {
        .size = sizeof(BMPInfoHeader),
        .width = img.cols,
        .height = img.rows,
        .planes = 1,
        .bit_count = 24,
        .compression = 0,
        .image_size = image_size,
    };
    
    // 分配内存并转换图像数据，将RGB转换为BGR并限制值范围
    unsigned char* bmp_data = (unsigned char*)malloc(image_size);
    memset(bmp_data, 0, image_size);
    
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            int src_idx = (y * img.cols + x) * img.channels;
            // BMP是自下而上存储，所以需要y轴翻转
            int dst_idx = (img.rows - 1 - y) * row_size + x * 3;
            
            float r = img.data[src_idx + 2] * 255.0f;
            float g = img.data[src_idx + 1] * 255.0f;
            float b = img.data[src_idx] * 255.0f;
            
            // 确保像素值在0-255范围内
            bmp_data[dst_idx] = (unsigned char)(b < 0 ? 0 : (b > 255 ? 255 : b));
            bmp_data[dst_idx + 1] = (unsigned char)(g < 0 ? 0 : (g > 255 ? 255 : g));
            bmp_data[dst_idx + 2] = (unsigned char)(r < 0 ? 0 : (r > 255 ? 255 : r));
        }
    }
    
    // 写入文件头和像素数据
    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "无法创建文件: %s\n", path);
        free(bmp_data);
        return;
    }
    
    fwrite(&file_header, sizeof(BMPFileHeader), 1, file);
    fwrite(&info_header, sizeof(BMPInfoHeader), 1, file);
    fwrite(bmp_data, 1, image_size, file);
    fclose(file);
    
    free(bmp_data);
}

// 图像预处理函数，包括调整大小和归一化
Matrix preprocess_image(Matrix img) {
    // 调整大小到模型输入尺寸 168x48
    Matrix resized = create_matrix(48, 168, img.channels);
    
    // 使用简化的双线性插值进行图像缩放
    float x_ratio = (float)(img.cols - 1) / 168;
    float y_ratio = (float)(img.rows - 1) / 48;
    
    for (int y = 0; y < 48; y++) {
        for (int x = 0; x < 168; x++) {
            // 计算原始图像中对应的四个像素点
            int x_low = (int)(x * x_ratio);
            int y_low = (int)(y * y_ratio);
            int x_high = x_low + 1;
            int y_high = y_low + 1;
            
            // 计算插值权重
            float x_weight = (x * x_ratio) - x_low;
            float y_weight = (y * y_ratio) - y_low;
            
            for (int c = 0; c < img.channels; c++) {
                // 获取四个点的像素值
                float a = img.data[(y_low * img.cols + x_low) * img.channels + c];
                float b = img.data[(y_low * img.cols + x_high) * img.channels + c];
                float c_val = img.data[(y_high * img.cols + x_low) * img.channels + c];
                float d = img.data[(y_high * img.cols + x_high) * img.channels + c];
                
                // 进行双线性插值
                float value = a * (1 - x_weight) * (1 - y_weight) +
                              b * x_weight * (1 - y_weight) +
                              c_val * (1 - x_weight) * y_weight +
                              d * x_weight * y_weight;
                
                resized.data[(y * 168 + x) * resized.channels + c] = value;
            }
        }
    }
    
    // 对图像进行归一化，使用预设的均值和标准差
    for (int i = 0; i < 48 * 168 * 3; i++) {
        resized.data[i] = (resized.data[i] - mean_value) / std_value;
    }
    
    return resized;
}

// ---
// 模型层结构体和前向传播函数

// 卷积层结构体
typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    Matrix weights; // 卷积核权重
    Matrix bias;    // 偏置项
} ConvLayer;

// 创建卷积层，并随机初始化权重和偏置（注意：实际模型应从已训练的文件中加载）
ConvLayer create_conv_layer(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    ConvLayer layer;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_size = kernel_size;
    layer.stride = stride;
    layer.padding = padding;
    
    int weights_size = out_channels * in_channels * kernel_size * kernel_size;
    layer.weights = create_matrix(1, weights_size, 1);
    
    // 随机初始化权重，范围为 -1 到 1
    for (int i = 0; i < weights_size; i++) {
        layer.weights.data[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
    
    layer.bias = create_matrix(1, out_channels, 1);
    for (int i = 0; i < out_channels; i++) {
        layer.bias.data[i] = 0.0f; // 偏置初始化为0
    }
    
    return layer;
}

// 卷积操作的前向传播
Matrix conv2d(Matrix input, ConvLayer layer) {
    // 计算输出特征图的尺寸
    int output_height = (input.rows + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
    int output_width = (input.cols + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
    
    Matrix output = create_matrix(output_height, output_width, layer.out_channels);
    
    // 遍历输出特征图的每个像素
    for (int out_c = 0; out_c < layer.out_channels; out_c++) {
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                float sum = 0.0f;
                
                // 遍历输入通道和卷积核
                for (int in_c = 0; in_c < layer.in_channels; in_c++) {
                    for (int ky = 0; ky < layer.kernel_size; ky++) {
                        for (int kx = 0; kx < layer.kernel_size; kx++) {
                            // 计算输入图像中对应的像素位置（考虑步长和填充）
                            int input_y = y * layer.stride + ky - layer.padding;
                            int input_x = x * layer.stride + kx - layer.padding;
                            
                            // 检查像素位置是否在有效范围内
                            if (input_y >= 0 && input_y < input.rows && input_x >= 0 && input_x < input.cols) {
                                int input_idx = (input_y * input.cols + input_x) * input.channels + in_c;
                                int weight_idx = ((out_c * layer.in_channels + in_c) * layer.kernel_size + ky) * layer.kernel_size + kx;
                                
                                sum += input.data[input_idx] * layer.weights.data[weight_idx];
                            }
                        }
                    }
                }
                
                // 加上偏置项
                int output_idx = (y * output_width + x) * output.channels + out_c;
                output.data[output_idx] = sum + layer.bias.data[out_c];
            }
        }
    }
    
    return output;
}

// ReLU激活函数，将所有负值设置为0
void relu(Matrix* mat) {
    for (int i = 0; i < mat->rows * mat->cols * mat->channels; i++) {
        if (mat->data[i] < 0) {
            mat->data[i] = 0;
        }
    }
}

// 最大池化操作，提取每个池化窗口的最大值
Matrix max_pool2d(Matrix input, int pool_size, int stride) {
    // 计算输出尺寸
    int output_height = (input.rows - pool_size) / stride + 1;
    int output_width = (input.cols - pool_size) / stride + 1;
    
    Matrix output = create_matrix(output_height, output_width, input.channels);
    
    // 遍历每个通道
    for (int c = 0; c < input.channels; c++) {
        // 遍历输出特征图的每个位置
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                float max_val = -1e10f; // 初始化一个非常小的数
                
                // 遍历池化窗口
                for (int py = 0; py < pool_size; py++) {
                    for (int px = 0; px < pool_size; px++) {
                        int input_y = y * stride + py;
                        int input_x = x * stride + px;
                        
                        // 检查边界
                        if (input_y < input.rows && input_x < input.cols) {
                            int input_idx = (input_y * input.cols + input_x) * input.channels + c;
                            if (input.data[input_idx] > max_val) {
                                max_val = input.data[input_idx];
                            }
                        }
                    }
                }
                
                int output_idx = (y * output_width + x) * output.channels + c;
                output.data[output_idx] = max_val;
            }
        }
    }
    
    return output;
}

// 全连接层结构体
typedef struct {
    int in_features;  // 输入特征数
    int out_features; // 输出特征数
    Matrix weights;
    Matrix bias;
} LinearLayer;

// 创建全连接层，并随机初始化权重和偏置
LinearLayer create_linear_layer(int in_features, int out_features) {
    LinearLayer layer;
    layer.in_features = in_features;
    layer.out_features = out_features;
    
    layer.weights = create_matrix(1, in_features * out_features, 1);
    for (int i = 0; i < in_features * out_features; i++) {
        layer.weights.data[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
    
    layer.bias = create_matrix(1, out_features, 1);
    for (int i = 0; i < out_features; i++) {
        layer.bias.data[i] = 0.0f;
    }
    
    return layer;
}

// 全连接层前向传播
Matrix linear_forward(Matrix input, LinearLayer layer) {
    Matrix output = create_matrix(1, layer.out_features, 1);
    
    for (int i = 0; i < layer.out_features; i++) {
        output.data[i] = 0.0f;
        for (int j = 0; j < layer.in_features; j++) {
            output.data[i] += input.data[j] * layer.weights.data[i * layer.in_features + j];
        }
        output.data[i] += layer.bias.data[i];
    }
    
    return output;
}

// Softmax函数，将输出值转换为概率分布
void softmax(Matrix* mat) {
    float max_val = -1e10f;
    float sum = 0.0f;
    
    // 找到最大值，用于数值稳定（避免e的次方数太大而溢出）
    for (int i = 0; i < mat->cols; i++) {
        if (mat->data[i] > max_val) {
            max_val = mat->data[i];
        }
    }
    
    // 计算指数和
    for (int i = 0; i < mat->cols; i++) {
        mat->data[i] = expf(mat->data[i] - max_val);
        sum += mat->data[i];
    }
    
    // 归一化，得到每个类别的概率
    for (int i = 0; i < mat->cols; i++) {
        mat->data[i] /= sum;
    }
}

// ---
// 完整的车牌识别模型结构和前向传播

// 车牌识别模型结构体
typedef struct {
    ConvLayer conv1;
    ConvLayer conv2;
    LinearLayer fc1;
    LinearLayer fc2;
} PlateRecModel;

// 初始化车牌识别模型，创建各个层
PlateRecModel create_plate_rec_model() {
    PlateRecModel model;
    
    // 第一层：输入3通道，输出32通道，3x3卷积核，步长1，填充1
    model.conv1 = create_conv_layer(3, 32, 3, 1, 1);
    // 第二层：输入32通道，输出64通道，3x3卷积核，步长1，填充1
    model.conv2 = create_conv_layer(32, 64, 3, 1, 1);
    // 全连接层1：将特征图展平为向量，输入64*21*5特征，输出128个特征
    model.fc1 = create_linear_layer(64 * 21 * 5, 128); 
    // 全连接层2（输出层）：输入128个特征，输出等于字符集大小的特征，用于分类
    model.fc2 = create_linear_layer(128, strlen(plate_chars));
    
    return model;
}

// 车牌识别模型的前向传播（推理）
Matrix plate_rec_forward(Matrix input, PlateRecModel model) {
    // 1. 卷积层1 + ReLU + 最大池化
    Matrix x = conv2d(input, model.conv1);
    relu(&x);
    x = max_pool2d(x, 2, 2);
    
    // 2. 卷积层2 + ReLU + 最大池化
    x = conv2d(x, model.conv2);
    relu(&x);
    x = max_pool2d(x, 2, 2);
    
    // 3. 将特征图展平为一维向量
    Matrix flattened = create_matrix(1, x.rows * x.cols * x.channels, 1);
    memcpy(flattened.data, x.data, x.rows * x.cols * x.channels * sizeof(float));
    free_matrix(&x);
    
    // 4. 全连接层1 + ReLU
    x = linear_forward(flattened, model.fc1);
    free_matrix(&flattened);
    relu(&x);
    
    // 5. 全连接层2（输出层）
    Matrix output = linear_forward(x, model.fc2);
    free_matrix(&x);
    
    // 6. Softmax激活，得到概率分布
    softmax(&output);
    
    return output;
}

// ---
// 识别结果解码和主函数

// 解码识别结果，这里是简化版，只取最高概率的字符
void decode_plate(Matrix output, char* plate, float* confidence) {
    int max_idx = 0;
    float max_val = 0.0f;
    
    // 找到概率最高的字符索引和其概率值
    for (int i = 0; i < output.cols; i++) {
        if (output.data[i] > max_val) {
            max_val = output.data[i];
            max_idx = i;
        }
    }
    
    *confidence = max_val;
    // 将最高概率的字符复制到结果字符串中
    plate[0] = plate_chars[max_idx];
    plate[1] = '\0';
}

// 车牌检测函数（简化版）：不进行真正的检测，只是假定车牌在图像中心区域
int detect_plate(Matrix image, int* x, int* y, int* width, int* height) {
    *x = image.cols / 4;
    *y = image.rows / 4;
    *width = image.cols / 2;
    *height = image.rows / 4;
    
    return 1; // 假设检测到一个车牌
}

// 主函数，程序的入口
int main() {
    // 初始化随机数种子，用于创建随机权重
    srand(time(NULL));
    
    // 创建一个包含随机权重的模型
    PlateRecModel model = create_plate_rec_model();
    
    // 读取测试图像
    Matrix image = read_bmp("test_plate.bmp");
    if (image.data == NULL) {
        printf("无法读取图像\n");
        return 1;
    }
    
    // 检测车牌位置（简化版）
    int x, y, width, height;
    int plate_count = detect_plate(image, &x, &y, &width, &height);
    
    if (plate_count > 0) {
        // 裁剪出车牌区域
        Matrix plate_region = create_matrix(height, width, image.channels);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int c = 0; c < image.channels; c++) {
                    int src_idx = ((y + i) * image.cols + (x + j)) * image.channels + c;
                    int dst_idx = (i * width + j) * plate_region.channels + c;
                    plate_region.data[dst_idx] = image.data[src_idx];
                }
            }
        }
        
        // 对裁剪后的图像进行预处理
        Matrix processed = preprocess_image(plate_region);
        free_matrix(&plate_region);
        
        // 将预处理后的图像输入模型进行推理
        Matrix output = plate_rec_forward(processed, model);
        free_matrix(&processed);
        
        // 解码模型的输出结果
        char plate[16];
        float confidence;
        decode_plate(output, plate, &confidence);
        free_matrix(&output);
        
        printf("识别结果: %s (置信度: %.2f)\n", plate, confidence);
        
        // 在原始图像上绘制红色边框
        // 绘制顶部和底部边框
        for (int i = 0; i < width; i++) {
            // 设置颜色为红色（B=0, G=0, R=1）
            image.data[(y * image.cols + (x + i)) * image.channels] = 0;
            image.data[(y * image.cols + (x + i)) * image.channels + 1] = 0;
            image.data[(y * image.cols + (x + i)) * image.channels + 2] = 1.0f;
            
            image.data[((y + height - 1) * image.cols + (x + i)) * image.channels] = 0;
            image.data[((y + height - 1) * image.cols + (x + i)) * image.channels + 1] = 0;
            image.data[((y + height - 1) * image.cols + (x + i)) * image.channels + 2] = 1.0f;
        }
        
        // 绘制左侧和右侧边框
        for (int i = 0; i < height; i++) {
            image.data[((y + i) * image.cols + x) * image.channels] = 0;
            image.data[((y + i) * image.cols + x) * image.channels + 1] = 0;
            image.data[((y + i) * image.cols + x) * image.channels + 2] = 1.0f;
            
            image.data[((y + i) * image.cols + (x + width - 1)) * image.channels] = 0;
            image.data[((y + i) * image.cols + (x + width - 1)) * image.channels + 1] = 0;
            image.data[((y + i) * image.cols + (x + width - 1)) * image.channels + 2] = 1.0f;
        }
    }
    
    // 保存带边框的图像到新文件
    save_bmp("result.bmp", image);
    free_matrix(&image);
    
    // 释放模型占用的所有内存
    free_matrix(&model.conv1.weights);
    free_matrix(&model.conv1.bias);
    free_matrix(&model.conv2.weights);
    free_matrix(&model.conv2.bias);
    free_matrix(&model.fc1.weights);
    free_matrix(&model.fc1.bias);
    free_matrix(&model.fc2.weights);
    free_matrix(&model.fc2.bias);
    
    return 0;
}