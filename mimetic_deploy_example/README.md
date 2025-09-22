# 文件功能说明
model_train.py: 训练手写数字识别网络的python代码
model.pt: 经由model_train.py训练得到的模型权重
model_weights_export.py: 将model.pt权重导出为c代码可读取的.txt文件
model_weights.txt: 经由上一步导出的.txt格式的模型权重
model.c: 由C代码定义的手写数字识别网络，包括了读取权重，执行推理，展示结果的全流程

# 使用方法
$python model_train.py            // 训练模型
$python model_weights_export.py   // 导出模型权重为txt
$gcc model.c -o model -lm         // C编译
$./model                          // 运行模型