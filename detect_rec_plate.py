import torch
import cv2
import numpy as np
import argparse
import copy
import time
import os
import torch.onnx as onnx
from shapely.geometry import Polygon, box
import sys
import io

from ptflops import get_model_complexity_info
from ultralytics.nn.tasks import attempt_load_weights
from plate_recognition.plate_rec import get_plate_result, init_model
from fonts.cv_puttext import cv2ImgAddText
from plate_recognition.double_plate_split_merge import get_split_merge

# === CCPD 注释规则字符映射表 ===
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

def parse_ccpd_filename(filename):
    """
    根据CCP-DB文件名格式，解析出车牌号的地面真值。
    此版本修复了对8位编码文件名的错误处理。
    """
    try:
        parts = os.path.basename(filename).split('-')
        if len(parts) < 3:
            print(f"警告: 文件名格式不正确，缺少足够的 '-' 分隔符: {filename}")
            return None

        plate_char_codes = parts[-3].split('_')
        
        # 验证编码列表长度是否至少为3 (省份+字母+至少一个数字)
        if len(plate_char_codes) < 3:
            print(f"警告: 编码列表长度不足3，无法解析车牌号: {plate_char_codes}")
            return None

        # 汉字：第一个字符
        province_code = int(plate_char_codes[0])
        province = PROVINCES[province_code]
        
        # 字母：第二个字符
        alphabet_code = int(plate_char_codes[1])
        alphabet = ALPHABETS[alphabet_code]
        
        # 后续所有字符：使用ADS数组
        rest_of_plate = ""
        # 循环从第三个字符开始，到列表的末尾
        for i in range(2, len(plate_char_codes)):
            char_code = int(plate_char_codes[i])
            char = ADS[char_code]
            rest_of_plate += char
            
        plate_number = f"{province}{alphabet}{rest_of_plate}"
        return plate_number
    except (IndexError, ValueError) as e:
        print(f"警告: 文件名格式不正确或解析错误: {e}, 文件: {filename}")
        return None

def allFilePath(rootPath, allFIleList):
    """Recursively gets all file paths in a folder."""
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)

def four_point_transform(image, pts):
    """Corrects an irregular quadrilateral area into a standard rectangular image."""
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def letter_box(img, size=(640, 640), timer_dict=None):
    """Resizes and pads an image to a unified size with timing."""
    h, w, _ = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    
    start_resize = time.time()
    new_img = cv2.resize(img, (new_w, new_h))
    if timer_dict: timer_dict['cv2.resize'] += time.time() - start_resize
    
    left = int((size[1] - new_w) / 2)
    top = int((size[0] - new_h) / 2)
    right = size[1] - left - new_w
    bottom = size[0] - top - new_h
    
    start_border = time.time()
    img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if timer_dict: timer_dict['cv2.copyMakeBorder'] += time.time() - start_border
    
    return img, r, left, top

def load_model(weights, device):
    """Loads weights and calculates model parameters and FLOPs."""
    model = attempt_load_weights(weights, device=device)
    return model

def print_model_stats(model, device):
    """Prints detailed FLOPs and parameter stats using ptflops."""
    print("-" * 80)
    print("模型总计算量与总参数量统计报告 (基于 ptflops)")
    print("-" * 80)
    flops, params = get_model_complexity_info(
        model,
        (3, 640, 640),
        as_strings=True,
        print_per_layer_stat=False
    )
    print(f"模型总计算量 (FLOPs): {flops}")
    print(f"模型总参数量: {params}")
    print("-" * 80)

def xywh2xyxy(det):
    """Converts prediction from center-wh to top-left-bottom-right format."""
    y = det.clone()
    y[:, 0] = det[:, 0] - det[0:, 2] / 2
    y[:, 1] = det[:, 1] - det[0:, 3] / 2
    y[:, 2] = det[:, 0] + det[0:, 2] / 2
    y[:, 3] = det[:, 1] + det[0:, 3] / 2
    return y

def my_nums(dets, iou_thresh):
    """Performs Non-Maximum Suppression (NMS) to filter best bounding boxes."""
    y = dets.clone()
    y_box_score = y[:, :5]
    index = torch.argsort(y_box_score[:, -1], descending=True)
    keep = []
    while index.size()[0] > 0:
        i = index[0].item()
        keep.append(i)
        x1 = torch.maximum(y_box_score[i, 0], y_box_score[index[1:], 0])
        y1 = torch.maximum(y_box_score[i, 1], y_box_score[index[1:], 1])
        x2 = torch.minimum(y_box_score[i, 2], y_box_score[index[1:], 2])
        y2 = torch.minimum(y_box_score[i, 3], y_box_score[index[1:], 3])
        zero_ = torch.tensor(0).to(device)
        w = torch.maximum(zero_, x2 - x1)
        h = torch.maximum(zero_, y2 - y1)
        inter_area = w * h
        nuion_area1 = (y_box_score[i, 2] - y_box_score[i, 0]) * (y_box_score[i, 3] - y_box_score[i, 1])
        union_area2 = (y_box_score[index[1:], 2] - y_box_score[index[1:], 0]) * (y_box_score[index[1:], 3] - y_box_score[index[1:], 1])
        iou = inter_area / (nuion_area1 + union_area2 - inter_area)
        idx = torch.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(dets, r, left, top):
    """Restores bounding box coordinates to the original image size."""
    dets[:, [0, 2]] = dets[:, [0, 2]] - left
    dets[:, [1, 3]] = dets[:, [1, 3]] - top
    dets[:, :4] /= r
    return dets

def post_processing(prediction, conf, iou_thresh, r, left, top, timer_dict):
    """Filters, performs NMS, and restores coordinates for detection results with timing."""
    start_permute = time.time()
    prediction = prediction.permute(0, 2, 1).squeeze(0)
    timer_dict['permute_squeeze'] += time.time() - start_permute

    start_filter = time.time()
    xc = prediction[:, 4:6].amax(1) > conf
    x = prediction[xc]
    timer_dict['confidence_filter'] += time.time() - start_filter

    if not len(x):
        return []
    
    start_xywh2xyxy = time.time()
    boxes = x[:, :4]
    boxes = xywh2xyxy(boxes)
    timer_dict['xywh2xyxy'] += time.time() - start_xywh2xyxy
    
    start_cat = time.time()
    score, index = torch.max(x[:, 4:6], dim=-1, keepdim=True)
    x = torch.cat((boxes, score, x[:, 6:14], index), dim=1)
    timer_dict['torch.cat'] += time.time() - start_cat

    start_nms = time.time()
    keep = my_nums(x, iou_thresh)
    x = x[keep]
    timer_dict['my_nums(NMS)'] += time.time() - start_nms
    
    start_restore = time.time()
    x = restore_box(x, r, left, top)
    timer_dict['restore_box'] += time.time() - start_restore
    
    return x

def pre_processing(img, opt, device, timer_dict):
    """Pre-processes an image for inference with timing."""
    img, r, left, top = letter_box(img, (opt.img_size, opt.img_size), timer_dict)

    start_transpose = time.time()
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    timer_dict['transpose_copy'] += time.time() - start_transpose

    start_from_numpy = time.time()
    img = torch.from_numpy(img).to(device)
    timer_dict['from_numpy_to_torch'] += time.time() - start_from_numpy

    start_float_normalize = time.time()
    img = img.float() / 255.0
    timer_dict['float_and_normalize'] += time.time() - start_float_normalize

    start_unsqueeze = time.time()
    img = img.unsqueeze(0)
    timer_dict['unsqueeze'] += time.time() - start_unsqueeze
    
    return img, r, left, top

def is_illegal_parked(car_bbox, illegal_polygon):
    """Checks if a car's bounding box overlaps with an illegal parking area."""
    car_box = box(car_bbox[0], car_bbox[1], car_bbox[2], car_bbox[3])
    intersection_area = illegal_polygon.intersection(car_box).area
    return intersection_area > 0

def det_rec_plate(img, img_ori, detect_model, plate_rec_model, illegal_polygons, timer_dict):
    """Performs license plate detection and recognition with timing."""
    result_list = []
    
    start_preprocess = time.time()
    pre_timer_dict = timer_dict['pre_op_times']
    img, r, left, top = pre_processing(img, opt, device, pre_timer_dict)
    timer_dict['preprocessing'] += time.time() - start_preprocess

    start_detect_infer = time.time()
    predict = detect_model(img)[0]
    timer_dict['detect_inference'] += time.time() - start_detect_infer

    start_postprocess = time.time()
    post_timer_dict = timer_dict['post_op_times']
    outputs = post_processing(predict, 0.3, 0.5, r, left, top, post_timer_dict)
    timer_dict['postprocessing'] += time.time() - start_postprocess
    
    for output in outputs:
        result_dict = {}
        output = output.squeeze().cpu().numpy().tolist()
        rect = output[:4]
        rect = [int(x) for x in rect]
        label = output[-1]
        
        is_illegal = False
        for poly in illegal_polygons:
            if is_illegal_parked(rect, poly):
                is_illegal = True
                break
        
        y1, y2 = max(0, rect[1]), min(img_ori.shape[0], rect[3])
        x1, x2 = max(0, rect[0]), min(img_ori.shape[1], rect[2])

        if y2 <= y1 or x2 <= x1:
            print(f"警告: 裁剪出的车牌区域为空，跳过此检测结果。裁剪框坐标: {rect}")
            continue

        roi_img = img_ori[y1:y2, x1:x2]
        
        if int(label):
            roi_img = get_split_merge(roi_img)
        
        start_rec_infer = time.time()
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model, is_color=True)
        timer_dict['rec_inference'] += time.time() - start_rec_infer
        
        if not plate_number:
            print("警告: 车牌识别失败，跳过。")
            continue

        result_dict['plate_no'] = plate_number
        result_dict['plate_color'] = plate_color
        result_dict['rect'] = rect
        result_dict['detect_conf'] = output[4]
        result_dict['roi_height'] = roi_img.shape[0]
        result_dict['color_conf'] = color_conf
        result_dict['plate_type'] = int(label)
        result_dict['illegal_parked'] = is_illegal
        result_list.append(result_dict)
    return result_list

def draw_result(orgimg, dict_list, timer_dict, is_color=False):
    """Draws detection and recognition results on the image with timing."""
    illegal_plates = []
    
    start_rect = time.time()
    for result in dict_list:
        cv2.rectangle(orgimg, (result['rect'][0], result['rect'][1]), (result['rect'][2], result['rect'][3]), (0, 255, 0), 2)
    timer_dict['cv2.rectangle'] += time.time() - start_rect
    
    start_text = time.time()
    for result in dict_list:
        text_label = f"违停:{result['plate_no']}"
        labelSize = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        orgimg = cv2ImgAddText(orgimg, text_label, result['rect'][0], int(result['rect'][1] - round(1.6 * labelSize[0][1])), (0, 0, 0), 21)
        if result['illegal_parked']:
            illegal_plates.append(result['plate_no'])
    timer_dict['text_rendering'] += time.time() - start_text
    
    start_poly = time.time()
    for polygon in illegal_areas_points:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(orgimg, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    timer_dict['cv2.polylines'] += time.time() - start_poly

    return orgimg, illegal_plates

def save_model_as_onnx(model, model_name, device):
    """Exports a model to ONNX format."""
    print(f"Exporting model to {model_name}.onnx...")
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    onnx_path = f"{model_name}.onnx"
    onnx.export(model,
                dummy_input,
                onnx_path,
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}})
    return True, onnx_path

def save_rec_model_as_onnx(rec_model, model_name, device):
    """Exports a license plate recognition model to ONNX format."""
    print(f"Exporting recognition model to {model_name}.onnx...")
    dummy_input = torch.randn(1, 3, 48, 168).to(device)
    onnx_path = f"{model_name}.onnx"
    onnx.export(rec_model,
                dummy_input,
                onnx_path,
                verbose=False,
                input_names=['rec_input'],
                output_names=['rec_output'])
    return True, onnx_path

def print_model_per_layer_stats(model, device, input_size=(3, 640, 640)):
    """Prints per-layer FLOPs and parameter stats using ptflops."""
    print("-" * 80)
    print("模型逐层计算量与参数统计报告 (基于 ptflops)")
    print("-" * 80)
    with io.StringIO() as ost:
        get_model_complexity_info(
            model,
            input_size,
            as_strings=True,
            print_per_layer_stat=True,
            ost=ost
        )
        detailed_report = ost.getvalue()
    print(detailed_report)
    print("-" * 80)

def print_timing_report(total_preprocessing_time, total_detect_inference_time, total_postprocessing_time,
                          total_rec_inference_time, total_draw_time, total_pics):
    """Prints a summary of the average time spent on each task."""
    print("\n--- 任务环节平均耗时报告 ---")
    print(f"预处理平均耗时: {total_preprocessing_time / total_pics:.4f} s")
    print(f"检测模型推理平均耗时: {total_detect_inference_time / total_pics:.4f} s")
    print(f"后处理平均耗时: {total_postprocessing_time / total_pics:.4f} s")
    print(f"车牌识别模型推理平均耗时: {total_rec_inference_time / total_pics:.4f} s")
    print(f"结果绘制平均耗时: {total_draw_time / total_pics:.4f} s")

def print_pre_timing_report(pre_op_times, total_pics):
    print("\n--- 预处理环节平均耗时明细 ---")
    print(f"   - 图像缩放 (cv2.resize): {pre_op_times['cv2.resize'] / total_pics:.6f} s")
    print(f"   - 图像填充 (cv2.copyMakeBorder): {pre_op_times['cv2.copyMakeBorder'] / total_pics:.6f} s")
    print(f"   - 维度转换与复制 (transpose/copy): {pre_op_times['transpose_copy'] / total_pics:.6f} s")
    print(f"   - NumPy转PyTorch张量: {pre_op_times['from_numpy_to_torch'] / total_pics:.6f} s")
    print(f"   - 数据类型转换与归一化: {pre_op_times['float_and_normalize'] / total_pics:.6f} s")
    print(f"   - 增加批次维度 (unsqueeze): {pre_op_times['unsqueeze'] / total_pics:.6f} s")
    print("-" * 40)

def print_post_timing_report(post_op_times, total_pics):
    print("\n--- 后处理环节平均耗时明细 ---")
    print(f"   - 维度变换 (permute/squeeze): {post_op_times['permute_squeeze'] / total_pics:.6f} s")
    print(f"   - 置信度过滤: {post_op_times['confidence_filter'] / total_pics:.6f} s")
    print(f"   - 坐标格式转换 (xywh2xyxy): {post_op_times['xywh2xyxy'] / total_pics:.6f} s")
    print(f"   - 张量拼接 (torch.cat): {post_op_times['torch.cat'] / total_pics:.6f} s")
    print(f"   - NMS核心算法 (my_nums): {post_op_times['my_nums(NMS)'] / total_pics:.6f} s")
    print(f"   - 坐标还原 (restore_box): {post_op_times['restore_box'] / total_pics:.6f} s")
    print("-" * 40)

def print_draw_timing_report(draw_op_times, total_pics):
    print("\n--- 结果绘制环节平均耗时明细 ---")
    print(f"   - 绘制矩形 (cv2.rectangle): {draw_op_times['cv2.rectangle'] / total_pics:.6f} s")
    print(f"   - 文本处理与绘制 (cv2.getTextSize/cv2ImgAddText): {draw_op_times['text_rendering'] / total_pics:.6f} s")
    print(f"   - 绘制多边形 (cv2.polylines): {draw_op_times['cv2.polylines'] / total_pics:.6f} s")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'weights/yolov8s.pt', help='model.pt path(s)')
    parser.add_argument('--rec_model', type=str, default=r'weights/plate_rec_color.pth', help='model.pt path(s)')
    parser.add_argument('--image_path', type=str, default=r'ccpd_challenge', help='source')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='image_output', help='source')
    
    device = torch.device("cpu")
    
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    opt = parser.parse_args()
    save_path = opt.output

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    illegal_areas_points = [
        [(0, 0), (715, 0), (715, 763), (0, 763)],
    ]
    illegal_polygons = [Polygon(points) for points in illegal_areas_points]

    detect_model = load_model(opt.detect_model, device)
    plate_rec_model = init_model(device, opt.rec_model, is_color=True)

    if sum(p.numel() for p in detect_model.parameters()) == 0:
        print(f"\n错误: 检测模型 '{opt.detect_model}' 未成功加载，参数量为0。请检查文件路径和文件完整性。")
        sys.exit()
    if sum(p.numel() for p in plate_rec_model.parameters()) == 0:
        print(f"\n错误: 识别模型 '{opt.rec_model}' 未成功加载，参数量为0。请检查文件路径和文件完整性。")
        sys.exit()

    success_det, onnx_det_path = save_model_as_onnx(detect_model, "yolov8s_exported", device)
    if success_det:
        print(f"YOLOv8 ONNX file successfully generated at: {os.path.abspath(onnx_det_path)}")
    
    success_rec, onnx_rec_path = save_rec_model_as_onnx(plate_rec_model, "plate_rec_exported", device)
    if success_rec:
        print(f"Plate recognition ONNX file successfully generated at: {os.path.abspath(onnx_rec_path)}")

    detect_model.eval()
    file_list = []
    allFilePath(opt.image_path, file_list)
    count = 0
    
    total_preprocessing_time = 0
    total_detect_inference_time = 0
    total_postprocessing_time = 0
    total_rec_inference_time = 0
    total_draw_time = 0

    total_pre_op_times = {
        'cv2.resize': 0, 'cv2.copyMakeBorder': 0, 'transpose_copy': 0, 
        'from_numpy_to_torch': 0, 'float_and_normalize': 0, 'unsqueeze': 0
    }
    total_post_op_times = {
        'permute_squeeze': 0, 'confidence_filter': 0, 'xywh2xyxy': 0, 
        'torch.cat': 0, 'my_nums(NMS)': 0, 'restore_box': 0
    }
    total_draw_op_times = {
        'cv2.rectangle': 0, 'text_rendering': 0, 'cv2.polylines': 0
    }

    time_begin = time.time()
    all_illegal_plates = []
    
    correct_predictions = 0
    total_predictions = 0
    
    for pic_ in file_list:
        # === 将日志输出提前到处理每张图片之前 ===
        print(f"\n{count} {pic_}", end=" ")
        
        ground_truth_plate = parse_ccpd_filename(pic_)
        
        if ground_truth_plate:
            total_predictions += 1
            print(f"地面真值: {ground_truth_plate}", end=" ")
        
        img = cv2.imread(pic_)
        if img is None:
            print(f"警告: 无法读取图片 {pic_}，跳过。")
            continue

        img_ori = copy.deepcopy(img)
        
        timer_dict = {
            'preprocessing': 0, 'detect_inference': 0, 'postprocessing': 0, 
            'rec_inference': 0, 'pre_op_times': total_pre_op_times, 
            'post_op_times': total_post_op_times
        }
        result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model, illegal_polygons, timer_dict)
        
        # === 立即打印识别结果并进行比对 ===
        recognized_plates = [r['plate_no'] for r in result_list]
        
        # 优化比对逻辑，只要识别结果中包含一个与地面真值匹配的，就认为是正确的
        is_correct = False
        if ground_truth_plate:
            for recognized_plate in recognized_plates:
                if recognized_plate.strip().upper() == ground_truth_plate.strip().upper():
                    correct_predictions += 1
                    is_correct = True
                    break
        
        if is_correct:
            print(" -> 识别结果与地面真值匹配，正确！")
        else:
            print(f" -> 识别结果 {recognized_plates} 与地面真值不匹配，错误！")

        start_draw = time.time()
        ori_img, illegal_plates_in_pic = draw_result(img, result_list, total_draw_op_times)
        total_draw_time += time.time() - start_draw
        
        all_illegal_plates.extend(illegal_plates_in_pic)
        
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, ori_img)
        
        total_preprocessing_time += timer_dict['preprocessing']
        total_detect_inference_time += timer_dict['detect_inference']
        total_postprocessing_time += timer_dict['postprocessing']
        total_rec_inference_time += timer_dict['rec_inference']

        if illegal_plates_in_pic:
            print(f"图片 {img_name} 中的违停车辆车牌: {illegal_plates_in_pic}")
        else:
            print(f"图片 {img_name} 中未发现违停车辆。")

        count += 1

    print(f"\nsumTime time is {time.time() - time_begin:.2f} s")
    
    unique_illegal_plates = sorted(list(set(all_illegal_plates)))
    if unique_illegal_plates:
        print("\n--- 所有违停车辆车牌号汇总 ---")
        for plate in unique_illegal_plates:
            print(plate)
    else:
        print("\n--- 所有图片中均未发现违停车辆 ---")

    print("\n\n" + "=" * 80)
    print("模型性能统计报告")
    print("=" * 80)
    print_model_stats(detect_model, device)

    print("\n\n" + "=" * 80)
    print("YOLOv8 检测模型逐层计算量与参数报告")
    print("=" * 80)
    print_model_per_layer_stats(detect_model, device, input_size=(3, opt.img_size, opt.img_size))

    print("\n\n" + "=" * 80)
    print("车牌识别模型逐层计算量与参数报告")
    print("=" * 80)
    print_model_per_layer_stats(plate_rec_model, device, input_size=(3, 48, 168))
    
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("yolov8 detect 总参数量: %.2fM, rec 总参数量: %.2fM" % (total / 1e6, total_1 / 1e6))
    
    if len(file_list) > 0:
        print("\n\n" + "=" * 80)
        print("任务环节平均耗时报告 (实际运行性能)")
        print("=" * 80)
        print_timing_report(total_preprocessing_time, total_detect_inference_time, total_postprocessing_time,
                            total_rec_inference_time, total_draw_time, len(file_list))
        
        print_pre_timing_report(total_pre_op_times, len(file_list))
        print_post_timing_report(total_post_op_times, len(file_list))
        print_draw_timing_report(total_draw_op_times, len(file_list))

    print("\n\n" + "=" * 80)
    print("车牌识别准确率报告")
    print("=" * 80)
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"总计处理了 {total_predictions} 个有标签的图像。")
        print(f"其中准确识别的数量为 {correct_predictions}。")
        print(f"最终识别准确率: {accuracy:.2f}%")
    else:
        print("未找到可用于准确率评估的标签数据。请确保图片文件名遵循 CCPD 数据集格式。")
    print("=" * 80)

    print("=" * 80)
    print(f"当前任务已在 {device.type.upper()} 上完成。")
    print("=" * 80)