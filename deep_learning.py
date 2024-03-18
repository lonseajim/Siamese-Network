from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: torch.nn.Module, criterion, optimizer,
          scheduler=None, num_epochs=1000, early_stop_epochs=0, model_path='checkpoints/best_model.pth', log=False):
    """
    Train deep learning model
    :param train_dataloader: 训练集
    :param val_dataloader: 验证集或者测试集
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 优化器学习率调度
    :param num_epochs: 循环次数
    :param early_stop_epochs: 提前停止训练的循环数（当前循环数-最好结果的循环数）
    :param model_path: 模型保存路径
    :param log: 是否打印中间过程
    :return:
    """
    if log is True:
        train_dataloader = tqdm(train_dataloader)
        val_dataloader = tqdm(val_dataloader)

    best_acc = 0.0
    best_epoch = 0
    for ep in range(num_epochs):
        if log is True:
            print('Epoch: {}/{} --------------------------'.format(ep + 1, num_epochs))
        model.train()
        for spectra, labels in train_dataloader:
            spectra, labels = spectra.to(device).float(), labels.to(device)
            outputs = model(spectra)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # 在验证集上计算准确率
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for spectra, labels in val_dataloader:
                spectra, labels = spectra.to(device).float(), labels.to(device)
                outputs = model(spectra)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            best_epoch = ep + 1
            torch.save(model.state_dict(), model_path)
            if log is False:
                print('Best accuracy is: {:.2f}%, epoch: {}'.format(best_acc, best_epoch))
        if log is True:
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(ep + 1, num_epochs, loss.item(), acc))
            print('Best accuracy is: {:.2f}%, epoch: {}'.format(best_acc, best_epoch))

        # 如果当前循环数 - 最好结果的循环数大于early_stop_epochs，则停止训练
        if early_stop_epochs > 0:
            if ep - best_epoch > early_stop_epochs:
                break


def test(model, model_path, test_dataloader):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    true_list = []
    predicted_list = []
    with torch.no_grad():
        for spectra, labels in test_dataloader:
            true_list.extend(labels.numpy())
            spectra, labels = spectra.to(device).float(), labels.to(device)
            outputs = model(spectra)
            _, predicted = torch.max(outputs.data, 1)
            predicted_list.extend(predicted.cpu().numpy())
    return true_list, predicted_list
