import torch
import numpy as np
from torchvision import transforms


def ImageStretching(image):
    channels = image.shape[2]  # 获取图像的波段数，即通道数
    band_list = []  # 用来存储每个波段处理后的数据

    # 对每个波段进行拉伸
    for i in range(channels):
        band_data = image[:,:,i]  # 获取当前波段的所有数据
        band_min = np.percentile(band_data, 2)  # 获取当前波段的 2% 分位数
        band_max = np.percentile(band_data, 98)  # 获取当前波段的 98% 分位数
        # 进行归一化，将数据拉伸到 [0, 1] 范围
        band_data = (band_data - band_min) / (band_max - band_min)
        band_list.append(band_data)  # 将处理后的波段数据添加到列表中

    # 将所有波段的结果合并成一个多维数组
    image_data = np.stack(band_list, axis=-1) # 将处理后的所有波段数据沿着最后一个轴（即通道维度）堆叠成一个三维数组，形成形状为 (height, width, channels) 的图像数据。
    image_data = np.clip(image_data, 0, 1)  # 将值限制在 [0, 1] 范围内

    # 将数据放大到 [0, 255] 范围，并转换为整数类型
    image_data = (image_data * 255).astype(np.uint8)

    return image_data  # 返回处理后的图像数据



def normlize3D(image,use_group=False,group_num=4):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    size = image.shape
    if size[2]!=3:
        image_norms = []

        for i in range(size[2]):
            image_slice3 = image[:,:,i,:,:]
            image_slice_norm = transform(image_slice3)
            image_norms.append(image_slice_norm.unsqueeze(2))
        image_norms = torch.cat(image_norms,dim=2)
        if use_group:
            image_norms = image_norms.unsqueeze(0)
            grouped_channels = []
            for start_channel in range(0,group_num):
                grouped_channels.append(np.arange(start_channel,(image_norms.shape[2]//group_num)*group_num,group_num))
            grouped_img = torch.cat([image_norms[:, :, channels, :, :] for channels in grouped_channels], dim=0)
            return grouped_img.cuda()
        else:
            return image_norms
