import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from resnet import Resnet18
from tqdm import tqdm

if __name__ == '__main__':
    batch_size = 32
    Epoch = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')

    cifar_train = datasets.CIFAR10('cifar', True,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor()]
                                   ),
                                   download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      transforms.ToTensor()]
                                  ),
                                  download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    # 设置模型
    model = Resnet18().to(device)
    # 设置损失函数
    criteon = nn.CrossEntropyLoss().to(device)
    # 设置优化器
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0
    for epoch in range(Epoch):
        # 模型训练
        with tqdm(total=cifar_train.__len__(), desc=f'Epoch {epoch + 1}/{Epoch}') as pbar:
            # 设置模型为训练模式
            model.train()
            # 导入数据进行训练
            for batchidx, (x, label) in enumerate(cifar_train):
                x, label = x.to(device), label.to(device)
                # 前向计算
                logits = model(x)
                # 计算损失
                loss = criteon(logits, label)
                # 反向传播
                opt.zero_grad()
                loss.backward()
                opt.step()
                # 进度条更新
                pbar.set_postfix(**{
                    'batch idx': batchidx,
                    'loss': loss.item()
                })
                pbar.update(1)
            print(f'\ntrain::epoch: {epoch}  loss: {loss.item()}')
        # 模型验证
        with tqdm(total=cifar_test.__len__(), desc=f'Epoch {epoch + 1}/{Epoch}') as pbar:
            correct = 0
            num = 0
            acc = 0
            model.eval()
            with torch.no_grad():
                for x, label in cifar_test:
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
            # 找到准确率最高的进行参数保存
            if acc > best_acc:
                best_acc = acc
                path = './best.pth'
                print(f'\nbest acc: {best_acc} parameter save path: {path}')
                torch.save(model.state_dict(), path)
