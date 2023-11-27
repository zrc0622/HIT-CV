import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import re

def rgb2gray(dir):
    if isinstance(dir, str):
        rgb_img = Image.open(dir)# 读取图像
        gray_img = np.array(rgb_img.convert('L'), dtype=np.int32)# 转为灰度图，再转为numpy数组
    else: 
        rgb_img = dir
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        gray_img = gray_img.astype(np.int32)
    return rgb_img, gray_img

# 高斯算子
def gaussian_filter(sigma, size):
    middle = (size + 1)/2
    gaussian_function = lambda x,y : (1/(2*np.pi*sigma**2)) * np.exp(-((x**2 + y**2)/(2*sigma**2)))
    gau_filter = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            gau_filter[i][j] = gaussian_function(i-middle+1, j-middle+1)
    return gau_filter / np.sum(gau_filter)

# 卷积操作
def conv2d(input, kernel, bias=0, stride=1, padding=0):
    # input可以是图像经数组化的二维矩阵，kernel为卷积核本身输入，函数应可计算图像尺寸，改写卷积核，偏置，步长和填充。
    
    # 输入图像和卷积核的维度
    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape

    # 计算输出图像的尺寸
    output_height = int(((input_height - kernel_height + 2 * padding) / stride) + 1)
    output_width = int(((input_width - kernel_width + 2 * padding) / stride) + 1)

    # 创建一个用于存储卷积结果的数组
    output = np.zeros((output_height, output_width))

    # 应用填充
    if padding > 0:
        input = np.pad(input, padding, mode='constant') # 填充常数，默认为0

    # 进行卷积操作
    for i in range(0, input_height - kernel_height + 1, stride):
        for j in range(0, input_width - kernel_width + 1, stride):
            output[i, j] = np.sum(input[i:i+kernel_height, j:j+kernel_width] * kernel) + bias

    return output

# 卷积滤波
def filter(img, kernel):
    process_img = conv2d(img, kernel, 0, 1, 1)

    min_pixel = np.min(process_img)
    max_pixel = np.max(process_img)
    stretched_image = ((process_img - min_pixel) / (max_pixel - min_pixel)) * 255
    # stretched_image = stretched_image.astype(np.uint8)  # 转换为0-255之间的整数

    return stretched_image

# CANNY边缘检测
def canny(gau_img):
    canny_img1 = np.array(gau_img)

    # 计算梯度
    grad_x = np.zeros_like(canny_img1)
    grad_y = np.zeros_like(canny_img1)

    grad_x[1:-1, 1:-1] = canny_img1[1:-1, 2:] + canny_img1[:-2, 2:] + 2 * canny_img1[1:-1, 1:-1] - \
                        canny_img1[:-2, :-2] - canny_img1[1:-1, :-2] - 2 * canny_img1[:-2, 1:-1]

    grad_y[1:-1, 1:-1] = canny_img1[2:, :-2] + canny_img1[2:, 2:] + 2 * canny_img1[2:, 1:-1] - \
                        canny_img1[:-2, :-2] - canny_img1[:-2, 2:] - 2 * canny_img1[:-2, 1:-1]

    grad_x = np.floor(grad_x / 4)
    grad_y = np.floor(grad_y / 4)

    grad_mag = np.floor(np.sqrt(grad_x**2 + grad_y**2))
    grad_mag[grad_mag > 255] = 255

    grad_dir = np.zeros_like(grad_x)
    grad_dir[grad_x == 0] = 2  # y 方向
    grad_dir[grad_x != 0] = np.floor(np.arctan2(grad_y[grad_x != 0], grad_x[grad_x != 0]) / (np.pi / 8))
    grad_dir[(grad_dir < 0) & (grad_dir >= -3)] += 4  # 将负角度转换为正值

    # 非极大值抑制
    canny_img2 = grad_mag.copy()

    for j in range(1, grad_mag.shape[0] - 1):
        for i in range(1, grad_mag.shape[1] - 1):
            dir = grad_dir[j, i]
            grad_now = grad_mag[j, i]

            if dir == 0 and (grad_now < grad_mag[j, i + 1] or grad_now < grad_mag[j, i - 1]):
                canny_img2[j, i] = 0
            elif dir == 1 and (grad_now < grad_mag[j + 1, i + 1] or grad_now < grad_mag[j - 1, i - 1]):
                canny_img2[j, i] = 0
            elif dir == 2 and (grad_now < grad_mag[j + 1, i] or grad_now < grad_mag[j - 1, i]):
                canny_img2[j, i] = 0
            elif dir == 3 and (grad_now < grad_mag[j + 1, i - 1] or grad_now < grad_mag[j - 1, i + 1]):
                canny_img2[j, i] = 0

    # 双阈值检测
    low = 10
    high = 20

    canny_img2[(canny_img2 >= low) & (canny_img2 <= high)] = 255
    canny_img2[canny_img2 < low] = 0


    # 抑制孤立的弱边缘
    listx, listy = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    canny_img3 = canny_img2.copy()

    for j in range(1, canny_img2.shape[0] - 1):
        for i in range(1, canny_img2.shape[1] - 1):
            if np.all(canny_img2[j + listy, i + listx] != 255):
                canny_img3[j, i] = 0

    return canny_img3

# 提取ROI
def mask(canny_img):
    # 定义ROI区域
    height, width = canny_img.shape
    roi_vertices = [(50, height), (width / 2-40, height / 2+30), (width / 2+40, height / 2+30), (width-50, height)]

    # 创建掩膜
    mask = np.zeros_like(canny_img)
    cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)

    # 应用掩膜
    masked_img = cv2.bitwise_and(canny_img, mask)

    return masked_img

# 画线
def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            if x1+x2+y1+y2 != 0:
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return lines_image

def make_points(image, average):
    try:
        # print(average)
        slope, y_int = average
        y1 = image.shape[0]
        #how long we want our lines to be --> 3/5 the size of the image
        y2 = int(y1 * (3/5))
        #determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])
    except TypeError:
        # 处理解包失败的情况
        # 在这里添加相应的代码，例如打印错误信息或执行其他操作
        return np.array([0,0,0,0])

# 平均线
def average(image, lines):
    left = []
    right = []
    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

# hough变化检测直线
def hough_lines(image, rho_resolution, theta_resolution, threshold):
    # 图像的高度和宽度
    height, width = image.shape[:2]

    # 极坐标参数空间的大小
    rho_max = np.sqrt(height**2 + width**2)
    rho_bins = int(2 * np.ceil(rho_max / rho_resolution)) + 1  # 向上取整并加1
    theta_bins = int(np.ceil(np.pi / theta_resolution))  # 向上取整

    # 构建累加器
    accumulator = np.zeros((rho_bins, theta_bins), dtype=np.uint8)

    # 遍历图像中的每个像素
    edges = np.argwhere(image != 0)
    for y, x in edges:
        for theta_idx in range(theta_bins):
            theta = theta_idx * theta_resolution
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int(np.round(rho / rho_resolution)) + rho_bins // 2
            accumulator[rho_idx, theta_idx] += 1

    # 根据阈值选取直线
    lines = []
    for rho_idx in range(rho_bins):
        for theta_idx in range(theta_bins):
            if accumulator[rho_idx, theta_idx] > threshold:
                rho = (rho_idx - rho_bins // 2) * rho_resolution
                theta = theta_idx * theta_resolution
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                lines.append(np.array([[x1, y1, x2, y2]]))

    return np.array(lines)

def process_img(image_dir, filter_size, filter_sigma, rho_resolution, theta_resolution, threshold):
    rgb_img, gray_img = rgb2gray(image_dir)
    gau_filter = gaussian_filter(filter_sigma, filter_size)
    gau_img = filter(gray_img, gau_filter) # 高斯模糊
    canny_img = canny(gau_img) # canny边缘检测
    mask_img = mask(canny_img) # 提取ROI
    mask_img_copy = mask_img.astype(np.uint8)
    rgb_img_copy = np.copy(rgb_img)
    rgb_img_copy = rgb_img_copy.astype(np.uint8)

    lines = hough_lines(mask_img_copy, rho_resolution, theta_resolution, threshold)

    averaged_lines = average(rgb_img_copy, lines)
    black_lines = display_lines(rgb_img_copy, averaged_lines)
    # lanes = cv2.addWeighted(rgb_img_copy, 1, black_lines, 1, 1)
    
    red_mask = (black_lines[:, :, 2] == 0) & (black_lines[:, :, 0] != 0) & (black_lines[:, :, 1] == 0)
    try:
        rgb_img_copy[red_mask] = [255, 0, 0]  # 设置为红色
    except:
        pass
    # plt.imsave('result_'+image_dir, rgb_img_copy)
    return rgb_img_copy

def extract_frames(video_path, output_dir, filter_size, filter_sigma, rho_resolution, theta_resolution, threshold):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    frame_count = 0

    while success:
        # 在这里添加车道检测的代码，对每一帧进行处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = process_img(rgb_frame, filter_size, filter_sigma, rho_resolution, theta_resolution, threshold)  # 自定义函数，对帧进行处理

        # 输出处理后的帧为图像文件
        output_path = f"{output_dir}/frame_{frame_count}.jpg"
        plt.imsave(output_path, processed_frame)

        success, frame = video.read()
        frame_count += 1

    video.release()

def images_to_video(input_folder, output_video_path, fps):
    image_files = os.listdir(input_folder)  # 获取文件夹下的所有文件名
    image_files = sorted(image_files, key=lambda x: int(re.findall(r'\d+', x)[0]))  # 按照数字的大小对文件名进行排序
    frame_size = None
    video_writer = None

    for image_file in image_files:
        if image_file.endswith('.jpg') or image_file.endswith('.png'):  # 修改为适合您的图片格式
            image_path = os.path.join(input_folder, image_file)
            frame = cv2.imread(image_path)  # 读取图片

            if frame_size is None:
                frame_size = (frame.shape[1], frame.shape[0])  # 获取第一张图片的尺寸

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要更改编码器
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

            video_writer.write(frame)  # 将帧写入视频

    if video_writer is not None:
        video_writer.release()  # 释放视频写入器

if __name__ == "__main__":
    # 图片视频参数
    image_dir = 'test_img.jpg'
    
    # 高斯滤波参数
    filter_size = 5
    filter_sigma = 0.8

    # hough参数
    rho_resolution = 1
    theta_resolution = np.pi / 180
    threshold = 100

    # 视频参数
    video_path = "test_video.mp4"
    output_dir = "video_images"
    output_video_path = "output_test_video.mp4"

    # process_img(image_dir, filter_size, filter_sigma, rho_resolution, theta_resolution, threshold)

    # 抽取帧并检测
    extract_frames(video_path, output_dir, filter_size, filter_sigma, rho_resolution, theta_resolution, threshold)
    # 合成视频
    images_to_video(output_dir, output_video_path)