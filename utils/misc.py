import numpy as np
import torch
from rich.progress import track

import random
import os
import time


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j

def get_padding(h, w, new_h, new_w):
    if h >= new_h:
        top = 0
        bottom = 0
    else:
        dh = new_h - h
        top = dh // 2
        bottom = dh // 2 + dh % 2
        h = new_h
    if w >= new_w:
        left = 0
        right = 0
    else:
        dw = new_w - w
        left = dw // 2
        right = dw // 2 + dw % 2
        w = new_w

    return (left, top, right, bottom), h, w

def cal_inner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area

def divide_img_into_patches(img, patch_size):
    h, w = img.shape[-2:]

    img_patches = []
    h_stride = int(np.ceil(1.0 * h / patch_size))
    w_stride = int(np.ceil(1.0 * w / patch_size))
    for i in range(h_stride):
        for j in range(w_stride):
            h_start = i * patch_size
            if i != h_stride - 1:
                h_end = (i + 1) * patch_size
            else:
                h_end = h
            w_start = j * patch_size
            if j != w_stride - 1:
                w_end = (j + 1) * patch_size
            else:
                w_end = w
            img_patches.append(img[..., h_start:h_end, w_start:w_end])

    return img_patches, h_stride, w_stride

def denormalize(img_tensor):
    # denormalize a image tensor
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
        img_tensor = img_tensor + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
    elif len(img_tensor.shape) == 4:
        img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
        img_tensor = img_tensor + torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
    # img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    # img_tensor = img_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    return img_tensor







def decoder_image(img, mean, std):  # 将图像进行解码，将其从标准化的值恢复到原始的像素值范围。输出为：解码后的图像。
    inputs_decoder = []
    for ss, m, s in zip(img, mean, std):
        ss = np.array(ss * s)
        ss = np.array(ss + m)
        ss = ss * 255
        inputs_decoder.append(ss)
    return np.stack(inputs_decoder)




def my_fft(img, threshold):  # 使用傅里叶变换处理图像，分离出低频和高频部分，并生成增强的图像。
    # 用于选择低频部分的阈值，如果未指定，会随机从指定的范围中选取。
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # print(img)
    img = decoder_image(img, mean, std)  # 首先对图像进行解码（通过decoder_image函数），将图像转换到正确的颜色空间。
    img = np.transpose(img, (1, 2, 0))  # 将图像转置为（H, W, C）的形式，其中H是高度，W是宽度，C是颜色通道数。

    H, W, C = img.shape

    if threshold == None:  # 如果用户没有传入 threshold 参数，随机从预设的 thresholds 列表中选择一个阈值。
        thresholds = [15, 30, 45, 60, 75, 90, 105, 120,135,150]
        index = np.random.randint(0, len(thresholds))
        threshold = thresholds[index]

    f = np.fft.fft2(img, axes=(0, 1))  # 对图像进行二维傅里叶变换（2D FFT）。np.fft.fft2 返回一个复数数组
    fshift = np.fft.fftshift(f)  # 使用 np.fft.fftshift 将频谱中的低频部分移动到频域的中心。这是因为傅里叶变换的结果通常会把低频部分放在图像的角落。

    '''上面一行这是为什么呢？'''  # 因为后面掩码保留的就是高频

    crows, ccols = int(H / 2), int(W / 2)  # 计算频域图像的中心位置，crows 和 ccols 分别是频域图像的中心行和中心列。

    mask = np.zeros((H, W, C), dtype=np.uint8)  # 创建一个与图像大小相同的全零掩码。掩码的大小与图像相同，但初始时是全零的。

    mask[crows - threshold:crows + threshold,
    ccols - threshold:ccols + threshold] = 1  # 求低频,使用给定的threshold值生成一个掩码，保留低频部分。
    '''具体操作是：在掩码中创建一个矩形区域，表示低频部分。在频谱的中心周围的 threshold 区域内设置掩码值为 1，其他部分为 0。'''

    fshift = fshift * mask  # 对频域图像应用掩码，以便保留低频部分，并去除高频部分。
    ishift = np.fft.ifftshift(fshift)  # 使用 np.fft.ifftshift 将频域图像的低频部分移回角落，准备进行反傅里叶变换。
    i_img = np.fft.ifft2(ishift, axes=(0, 1))  # 对掩蔽后的频域图像进行逆傅里叶变换，将其转换回空间域，得到低频部分的图像。
    i_img_L = np.abs(i_img)  # 获取低频图像的绝对值。由于逆傅里叶变换结果是复数，因此取其绝对值，得到图像的强度部分。

    img_H_temp = (img - i_img_L)  # 计算高频部分图像，方法是将原图像减去低频部分。这样剩下的就是图像的高频部分。
    
    max_value = np.max(img_H_temp)
    if max_value == 0:
       max_value = 1e-8  # 避免除以零
    img_H_temp = img_H_temp * (255 / max_value)* 3  # 归一化到 [0, 255]

    
    # img_H_temp = img_H_temp * (255 / np.max(img_H_temp)) * 3  # 将高频部分图像归一化，使其像素值范围保持在 0 到 255 之间，并通过乘以 3 增强高频部分的视觉效果。
    
    i_img_L[i_img_L>255] = 255
    i_img_L[i_img_L<0] = 0

    img_H_temp[img_H_temp > 255] = 255  # 对高频和低频部分图像进行裁剪，确保其像素值在有效范围 [0, 255] 内。
    img_H_temp[img_H_temp < 0] = 0

    img_H_temp = np.transpose(img_H_temp, (2, 0, 1))
    i_img_L = np.transpose(i_img_L, (2, 0, 1))
    return i_img_L,img_H_temp,threshold  # 频部分图像、高频部分图像、增强后的左右翻转图像。










def denormalize2(img_tensor):
    # denormalize a image tensor
    img_tensor = (img_tensor - img_tensor.min() / (img_tensor.max() - img_tensor.min()))
    return img_tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictAvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, n=1):
        for k, v in val.items():
            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_seeded_generator(seed):
    g = torch.Generator()
    g.manual_seed(0)
    return g

def get_current_datetime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def easy_track(iterable, description=None):
    return track(iterable, description=description, complete_style='dim cyan', total=len(iterable))
