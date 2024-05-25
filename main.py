import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import get_loader
from model import CharRNN, CharLSTM
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):

    model.train()
    total_loss = 0
    for input, target in tqdm(trn_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(input.size(0))
        output, hidden = model(input, hidden)
        loss = criterion(output.view(-1, output.size(2)), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input.size(0)
    trn_loss = total_loss / len(trn_loader.dataset)
    return trn_loss

def validate(model, val_loader, device, criterion):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, target in tqdm(val_loader):
            input, target = input.to(device), target.to(device)
            hidden = model.init_hidden(input.size(0))  # Batch size passed here
            output, hidden = model(input, hidden)
            loss = criterion(output.view(-1, output.size(2)), target.view(-1))
            total_loss += loss.item() * input.size(0)
    val_loss = total_loss / len(val_loader.dataset)
    return val_loss


def main():
    text_file_path = "shakespeare_train.txt"
    model_name = 'RNN'
    batch_size = 2048
    num_epochs = 50
    hidden_size = 64
    num_layers = 2
    
    train_loader, val_loader, size_ = get_loader(text_file_path, 
                                        batch_size = batch_size, 
                                        text_len = 30, 
                                        sub_sampling_ratio = 0.8)

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'RNN':
        model = CharRNN(size_, hidden_size, size_, num_layers=num_layers).to(device)
    elif model_name == 'LSTM':
        model = CharLSTM(size_, hidden_size, size_, num_layers=num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    trn_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Trn Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        trn_loss_list.append(trn_loss)
        val_loss_list.append(val_loss)
        
        #save the model with the best validation loss
        #save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_{}_model.pt'.format(model_name))
    #plotting best validation loss
    plt.figure(figsize=(8,6))
    sns.lineplot(trn_loss_list, marker='o', color='blue', label='training')
    sns.lineplot(val_loss_list, marker='o', color='orange', label='validation')

    plt.legend()
    plt.title(model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('train_{}.png'.format(model_name))
    plt.show()
if __name__ == '__main__':
    main()