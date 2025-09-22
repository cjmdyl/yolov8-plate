import torch
from ultralytics import YOLO

def print_yolov8_layer_info_direct(model_path):
    """
    加载 YOLOv8 模型并直接打印每一层的参数量。
    """
    try:
        # 加载 YOLO 模型
        model = YOLO(model_path)
        
        print("\n" + "=" * 80)
        print("YOLOv8 模型逐层参数量统计报告 (直接访问 PyTorch 模型)")
        print("=" * 80)
        
        # 获取核心 PyTorch 模型
        pt_model = model.model
        
        # 打印表头
        print(f"{'idx':<5} | {'模块名称':<30} | {'参数量':>15}")
        print("-" * 5 + "-" + "-" * 30 + "-" + "-" * 15)
        
        total_params = 0
        idx = 0
        
        # 遍历所有命名子模块并打印它们的参数量
        for name, module in pt_model.named_children():
            # 一个模块可能是一个单一的层或一个复合块（如 C2f）
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{idx:<5} | {name:<30} | {num_params:>15,}")
            total_params += num_params
            idx += 1
            
        print("-" * 80)
        print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
        print("-" * 80)

    except FileNotFoundError:
        print(f"\n错误: 找不到模型文件 '{model_path}'。请检查路径是否正确。")
    except Exception as e:
        print(f"\n加载模型或获取信息时出错: {e}")

if __name__ == '__main__':
    # 请确保这个路径指向您的 YOLOv8 模型文件
    yolov8_model_path = 'weights/yolov8m.pt'
    print_yolov8_layer_info_direct(yolov8_model_path)