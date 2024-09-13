from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: torch.nn.Module, criterion, optimizer,
          scheduler=None, num_epochs=1000, early_stop_epochs=0, model_path='checkpoints/best_model.pth', log=False):
    """
    Train deep learning model
    :param train_dataloader: train DataLoader
    :param val_dataloader: validation DataLoader
    :param model: model
    :param criterion: loss function
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param num_epochs: epoch
    :param early_stop_epochs: early stop epochs(current_epoch - best_model_epoch)
    :param model_path: best model save path
    :param log: log flag
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

        # test in validation DataLoader
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

        # if current_epoch - best_model_epoch > early_stop_epochs, stop train
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
