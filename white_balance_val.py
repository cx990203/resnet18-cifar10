# White balance validation
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List

from resnet import Resnet18
from tqdm import tqdm
from utils import *
import cv2
import numpy as np


def StyleChange(image: np.ndarray, change_mode=1) -> np.ndarray:
    if change_mode == 1:
        # 色调调整
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 0] = (image[:, :, 0] - 100) % 360
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 2:
        # 减小饱和度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 1] = image[:, :, 1] / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 3:
        # 减小亮度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 2] = image[:, :, 2] / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 4:
        # 增大饱和度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 1] = image[:, :, 1] + (1 - image[:, :, 1]) / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 5:
        # 增大亮度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 2] = image[:, :, 2] + (1 - image[:, :, 2]) / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 6:
        # 像素调整
        level = 10
        d = (1, 1, -1)      # 修正方向
        image[:, :, 0] = image[:, :, 0] + d[0] * level      # R
        image[:, :, 1] = image[:, :, 1] + d[1] * level      # G
        image[:, :, 2] = image[:, :, 2] - d[2] * level      # B
        image[image > 255] = 255
        image[image < 0] = 0

    return image


if __name__ == '__main__':
    # 设置运行参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1              # 一张一张图验证，只能设置为1
    para_path = './best-epoch=50.pth'       # 模型参数路径
    image_show_flag = False     # 显示输入图像标志位，如果显示，则会阻塞验证过程
    tone_change_flag = True     # 色调修改flag
    change_mode = 5             # 色调调整模式（只有启用了色调调整才能使用该变量修改模式）
    white_balance_method: List[str] = ['none', 'mean', 'perfect_reflective', 'grey_world', 'image_analysis', 'dynamic_threshold']       # 所有可以使用的白平衡方法
    white_balance_method_using: str = 'dynamic_threshold'      # 使用白平衡方法，如果不使用则为空''即可。只有开启了色调修改，白平衡方法才会生效
    # 加载数据集
    cifar_val = datasets.CIFAR10('cifar', False,
                                 transform=transforms.Compose([
                                     transforms.Resize((32, 32)),
                                     transforms.ToTensor()]
                                 ),
                                 download=True)
    cifar_val = DataLoader(cifar_val, batch_size=batch_size, shuffle=True)
    # 设置模型
    model = Resnet18().to(device)
    model.load_state_dict(torch.load(para_path, map_location=device))  # 导入模型参数
    model.eval()
    with tqdm(total=cifar_val.__len__(), desc=f'Validation') as pbar:
        correct = 0
        num = 0
        acc = 0
        model.eval()
        with torch.no_grad():
            for x, label in cifar_val:
                # 图片色调修正
                if tone_change_flag:
                    img = np.transpose(x.numpy()[0, :, :, :], [1, 2, 0])     # 将输入图像转换为numpy模式
                    img = (img * 255).astype(np.uint8)          # 图像逆归一化
                    img = StyleChange(img, change_mode=change_mode)       # 修改图像色调
                    if white_balance_method_using in white_balance_method:
                        # 图像白平衡处理
                        mode = white_balance_method.index(white_balance_method_using)       # 选择白平衡模式
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        img = white_balance(img, mode=mode, normal=False)             # 图像白平衡算法
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = normalize_input(img)          # 图像归一化
                    img = np.array([np.transpose(img, [2, 0, 1])])
                    x = torch.from_numpy(img).float()
                # 图片显示
                if image_show_flag:
                    # 显示输入图片
                    img = np.transpose(x.numpy()[0, :, :, :], [1, 2, 0])
                    cv2.namedWindow('val', cv2.WINDOW_NORMAL)
                    cv2.imshow('val', img)
                    cv2.waitKey(0)

                x, label = x.to(device), label.to(device)
                # 前向计算
                logits = model(x)
                pred = logits.argmax(dim=1)
                # 计算正确率（average）
                correct += torch.eq(pred, label).float().sum().item()
                num += x.size(0)
                acc = correct / num
                # 进度条更新
                pbar.set_postfix(**{
                    'correct': correct,
                    'num': num,
                    'avg acc': acc,
                })
                pbar.update(1)
    # 打印所有信息
    print('Validation finished!')
    print(f'using tone change:\033[0;31m {tone_change_flag} \033[0m')
    print(f'using white balance method:\033[0;31m {white_balance_method_using} \033[0m')
    print(f'Correct num/Sum:\033[0;31m {correct}/{num} \033[0m, Accuracy rate:\033[0;31m {acc} \033[0m')
