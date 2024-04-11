from model import LeNet5, CustomMLP
# import some packages you need here
from dataset import get_loader
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def plotting(data, y_label, state, epochs):
    epochs = [int(i+1) for i in range(epochs)]
    plt.plot(epochs, data, marker = 'o')
    plt.xlabel("Epoch")
    plt.ylabel(state + y_label)
    plt.title(state + y_label)
    plt.savefig("path your dir/plot/"+state +"_"+ y_label+".png")
    plt.clf()

def train(model, trn_loader, tst_loader, device, criterion, optimizer, epochs):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_accuracy = 0.0
        epoch_loss = 0.0
        correct = 0.0
        total_samples = 0
        for img, label in tqdm(trn_loader):
            optimizer.zero_grad()
            total_samples += label.size(0)
            
            img = img.to(device)
            label = label.to(device)
                        
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predict = torch.argmax(out, dim = -1)
            correct += torch.sum(predict == label).item()   
            
        epoch_accuracy = correct / total_samples
        # print(epoch_accuracy)
        train_acc.append(epoch_accuracy)
        
        epoch_loss = epoch_loss / len(trn_loader)
        train_loss.append(epoch_loss)
        # print(epoch_loss)
        
        eval_loss, eval_acc = test(model, tst_loader, device, criterion)
        test_acc.append(eval_acc)
        test_loss.append(eval_loss)
        # print(eval_loss)
        # print(eval_acc)
        
        print("{} epoch---loss:{}---accuracy:{}".format(epoch+1, round(epoch_loss, 3), round(epoch_accuracy, 3)))
        print("{} epoch---eval_loss:{}---eval_accuracy:{}".format(epoch+1, eval_loss, eval_acc))
        print("{} epoch complete!".format(epoch+1))
        print("="*60)
    return train_loss, train_acc, test_loss, test_acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model = model.to(device)
    model.eval()
    total_correct = 0.0
    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for img, label in tqdm(tst_loader):

            total_samples += label.size(0)
            
            img = img.to(device)
            label = label.to(device)
                        
            out = model(img)
            loss = criterion(out, label)
            
            total_loss += loss.item()
            predict = torch.argmax(out, dim = -1)
            correct += torch.sum(predict == label).item()   
        total_correct = correct / total_samples
        total_loss = total_loss / len(tst_loader)

        

    return round(total_loss, 3), round(total_correct, 3)


def main():
    """ Main function
        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss
    """
    # write your codes here
    device = "cpu"
    epochs = 20
    batch_size = 64
    
    train_dir = "path your dir"
    test_dir = "path your dir"
    
    LeNet5_train_loader = get_loader(train_dir, batch = batch_size, train_or_test = "train", model = "LeNet5")
    LeNet5_test_loader = get_loader(test_dir, batch = batch_size, train_or_test = "test", model = "LeNet5")
    
    CST_MLP_train_loader = get_loader(train_dir, batch = batch_size, train_or_test = "train", model = "CustomMLP")
    CST_MLPLeNet5_test_loader = get_loader(test_dir, batch = batch_size, train_or_test = "test", model = "CustomMLP")
    
    dense_img_size = len(CST_MLP_train_loader.dataset[0][0][-1])

    
    LeNet5_model = LeNet5(num_class=10)
    print("LeNet5 :", sum(p.numel() for p in LeNet5_model.parameters() if p.requires_grad))
    LeNet5_optimizer = torch.optim.SGD(LeNet5_model.parameters(), lr=0.01, momentum = 0.9)
    
    CustomMLP_model = CustomMLP(input_size = dense_img_size, num_class=10)
    print("CustomMLP_model :", sum(p.numel() for p in CustomMLP_model.parameters() if p.requires_grad))
    CustomMLP_optimizer = torch.optim.SGD(CustomMLP_model.parameters(), lr=0.01, momentum = 0.9)
    
    
    
    criterion = nn.CrossEntropyLoss()
    
    # LeNet5 training
    LeNet5_train_loss, LeNet5_train_acc, LeNet5_test_loss, LeNet5_test_acc = train(LeNet5_model, LeNet5_train_loader, LeNet5_test_loader, device, criterion, LeNet5_optimizer, epochs) # model, trn_loader, device, criterion, optimizer, epochs
    plotting(LeNet5_train_loss, "loss", "LeNet5_train", epochs)
    plotting(LeNet5_train_acc, "acc", "LeNet5_train", epochs)
    plotting(LeNet5_test_loss, "loss", "LeNet5_test", epochs)
    plotting(LeNet5_test_acc, "acc", "LeNet5_test", epochs)
    
    # CustomMLP_model training
    CustomMLP_train_loss, CustomMLP_train_acc, CustomMLP_test_loss, CustomMLP_test_acc = train(CustomMLP_model, CST_MLP_train_loader, CST_MLPLeNet5_test_loader, device, criterion, CustomMLP_optimizer, epochs) # model, trn_loader, device, criterion, optimizer, epochs
    plotting(CustomMLP_train_loss, "loss", "CustomMLP_train", epochs)
    plotting(CustomMLP_train_acc, "acc", "CustomMLP_train", epochs)
    plotting(CustomMLP_test_loss, "loss", "CustomMLP_test", epochs)
    plotting(CustomMLP_test_acc, "acc", "CustomMLP_test", epochs)

if __name__ == '__main__':
    main()
