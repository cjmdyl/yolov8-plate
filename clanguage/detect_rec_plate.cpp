#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 图像数据结构
typedef struct {
    int width;
    int height;
    int channels;
    unsigned char* data;
} Image;

// 车牌检测结果
typedef struct {
    int x, y, width, height;
    char plate_number[10];
} PlateDetection;

// 读取BMP图像（简单格式）
Image read_bmp(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("无法打开文件: %s\n", filename);
        exit(1);
    }
    
    // 读取BMP文件头
    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, file);
    
    // 提取图像尺寸信息
    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int channels = 3; // BMP通常是3通道
    
    // 分配内存
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data = (unsigned char*)malloc(width * height * channels);
    
    // 计算行填充字节
    int row_padded = (width * channels + 3) & (~3);
    unsigned char* row_data = (unsigned char*)malloc(row_padded);
    
    // 读取图像数据（BMP是倒置存储的）
    for (int y = height - 1; y >= 0; y--) {
        fread(row_data, sizeof(unsigned char), row_padded, file);
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * channels;
            int row_index = x * channels;
            img.data[index + 2] = row_data[row_index];     // B
            img.data[index + 1] = row_data[row_index + 1]; // G
            img.data[index] = row_data[row_index + 2];     // R
        }
    }
    
    free(row_data);
    fclose(file);
    return img;
}

// 保存BMP图像
void save_bmp(const char* filename, Image img) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("无法创建文件: %s\n", filename);
        return;
    }
    
    // BMP文件头
    unsigned char header[54] = {
        0x42, 0x4D,           // BM
        0, 0, 0, 0,           // 文件大小
        0, 0, 0, 0,           // 保留
        54, 0, 0, 0,          // 数据偏移
        40, 0, 0, 0,          // 信息头大小
        0, 0, 0, 0,           // 宽度
        0, 0, 0, 0,           // 高度
        1, 0,                 // 颜色平面数
        24, 0,                // 每像素位数
        0, 0, 0, 0,           // 压缩方式
        0, 0, 0, 0,           // 图像数据大小
        0, 0, 0, 0,           // 水平分辨率
        0, 0, 0, 0,           // 垂直分辨率
        0, 0, 0, 0,           // 使用的颜色数
        0, 0, 0, 0            // 重要颜色数
    };
    
    int row_padded = (img.width * 3 + 3) & (~3);
    int file_size = 54 + row_padded * img.height;
    
    *(int*)&header[2] = file_size;
    *(int*)&header[18] = img.width;
    *(int*)&header[22] = img.height;
    
    fwrite(header, sizeof(unsigned char), 54, file);
    
    // 写入图像数据（BMP是倒置存储的）
    unsigned char* row_data = (unsigned char*)malloc(row_padded);
    memset(row_data, 0, row_padded);
    
    for (int y = img.height - 1; y >= 0; y--) {
        for (int x = 0; x < img.width; x++) {
            int index = (y * img.width + x) * img.channels;
            int row_index = x * 3;
            
            row_data[row_index] = img.data[index + 2];     // B
            row_data[row_index + 1] = img.data[index + 1]; // G
            row_data[row_index + 2] = img.data[index];     // R
        }
        fwrite(row_data, sizeof(unsigned char), row_padded, file);
    }
    
    free(row_data);
    fclose(file);
}

// 转换为灰度图像
Image convert_to_grayscale(Image img) {
    Image gray;
    gray.width = img.width;
    gray.height = img.height;
    gray.channels = 1;
    gray.data = (unsigned char*)malloc(img.width * img.height);
    
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int color_index = (y * img.width + x) * img.channels;
            int gray_index = y * img.width + x;
            
            unsigned char r = img.data[color_index];
            unsigned char g = img.data[color_index + 1];
            unsigned char b = img.data[color_index + 2];
            
            gray.data[gray_index] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    
    return gray;
}

// 简单边缘检测（Sobel算子）
Image sobel_edge_detection(Image gray) {
    Image edges;
    edges.width = gray.width;
    edges.height = gray.height;
    edges.channels = 1;
    edges.data = (unsigned char*)malloc(gray.width * gray.height);
    memset(edges.data, 0, gray.width * gray.height);
    
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int y = 1; y < gray.height - 1; y++) {
        for (int x = 1; x < gray.width - 1; x++) {
            int gx = 0, gy = 0;
            
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int pixel = gray.data[(y + j) * gray.width + (x + i)];
                    gx += sobel_x[j + 1][i + 1] * pixel;
                    gy += sobel_y[j + 1][i + 1] * pixel;
                }
            }
            
            int magnitude = (int)sqrt(gx * gx + gy * gy);
            edges.data[y * gray.width + x] = magnitude > 255 ? 255 : magnitude;
        }
    }
    
    return edges;
}

// 简单车牌检测（基于颜色和形状特征）
PlateDetection detect_plate(Image img) {
    PlateDetection result = {0};
    
    // 转换为HSV颜色空间（简化版）
    Image hsv;
    hsv.width = img.width;
    hsv.height = img.height;
    hsv.channels = 3;
    hsv.data = (unsigned char*)malloc(img.width * img.height * 3);
    
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int index = (y * img.width + x) * 3;
            unsigned char r = img.data[index];
            unsigned char g = img.data[index + 1];
            unsigned char b = img.data[index + 2];
            
            // 简单RGB到HSV转换
            unsigned char max_val = fmax(fmax(r, g), b);
            unsigned char min_val = fmin(fmin(r, g), b);
            unsigned char v = max_val;
            unsigned char s = max_val == 0 ? 0 : (max_val - min_val) * 255 / max_val;
            
            hsv.data[index] = 0;     // H (简化处理)
            hsv.data[index + 1] = s; // S
            hsv.data[index + 2] = v; // V
        }
    }
    
    // 寻找蓝色区域（车牌的常见颜色）
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int index = (y * img.width + x) * 3;
            unsigned char r = img.data[index];
            unsigned char g = img.data[index + 1];
            unsigned char b = img.data[index + 2];
            
            // 简单蓝色检测
            if (b > r * 1.2 && b > g * 1.2 && b > 50) {
                // 标记为候选区域
                // 这里简化处理，实际需要更复杂的区域合并和验证
                if (result.width == 0) {
                    result.x = x;
                    result.y = y;
                    result.width = 1;
                    result.height = 1;
                } else {
                    // 扩展区域
                    if (x < result.x) {
                        result.width += result.x - x;
                        result.x = x;
                    }
                    if (y < result.y) {
                        result.height += result.y - y;
                        result.y = y;
                    }
                    if (x > result.x + result.width) {
                        result.width = x - result.x;
                    }
                    if (y > result.y + result.height) {
                        result.height = y - result.y;
                    }
                }
            }
        }
    }
    
    free(hsv.data);
    
    // 简单车牌号码识别（模拟）
    if (result.width > 0 && result.height > 0) {
        // 在实际应用中，这里应该实现字符分割和识别算法
        // 这里只是简单模拟返回一个固定值
        strcpy(result.plate_number, "ABC123");
    }
    
    return result;
}

// 在图像上绘制矩形
void draw_rectangle(Image img, int x, int y, int width, int height) {
    for (int i = x; i < x + width && i < img.width; i++) {
        for (int j = y; j < y + height && j < img.height; j++) {
            if (i == x || i == x + width - 1 || j == y || j == y + height - 1) {
                int index = (j * img.width + i) * img.channels;
                img.data[index] = 255;     // R
                img.data[index + 1] = 0;   // G
                img.data[index + 2] = 0;   // B
            }
        }
    }
}

// 主函数
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("使用方法: %s <输入图像> [输出图像]\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argc > 2 ? argv[2] : "output.bmp";
    
    // 读取图像
    printf("读取图像: %s\n", input_file);
    Image img = read_bmp(input_file);
    printf("图像尺寸: %dx%d, 通道数: %d\n", img.width, img.height, img.channels);
    
    // 检测车牌
    printf("检测车牌...\n");
    PlateDetection plate = detect_plate(img);
    
    if (plate.width > 0 && plate.height > 0) {
        printf("检测到车牌: x=%d, y=%d, width=%d, height=%d\n", 
               plate.x, plate.y, plate.width, plate.height);
        printf("车牌号码: %s\n", plate.plate_number);
        
        // 在图像上绘制车牌区域
        draw_rectangle(img, plate.x, plate.y, plate.width, plate.height);
    } else {
        printf("未检测到车牌\n");
    }
    
    // 保存结果
    printf("保存结果到: %s\n", output_file);
    save_bmp(output_file, img);
    
    // 释放内存
    free(img.data);
    
    return 0;
}