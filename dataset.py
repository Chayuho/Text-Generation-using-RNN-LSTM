# import some packages you need here
import pickle
from utils import create_vocab
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import tqdm as tqdm
import torch.nn.functional as F

# class Shakespeare(Dataset):

#     def __init__(self, total_text, text_len):
    
#         self.total_text = total_text
#         self.chars_set = sorted(set(self.total_text))
#         self.char_idx = {char: idx for idx, char in enumerate(self.chars_set)}
#         self.idx_char = {idx: char for char, idx in self.char_idx.items()}    
        
#         self.text_len = text_len
#         self.total_len = len(self.total_text) - self.text_len
        
#     def __len__(self):
        
#         return len(self.total_text)

#     def __getitem__(self, idx):
        
#         sequence = self.total_text[idx:idx + self.text_len]
#         sequence = [self.char_idx[char] for char in sequence]
#         sequence = F.one_hot(torch.LongTensor(sequence), num_classes=len(self.char_idx)).float()
        
#         label = self.total_text[idx + 1:idx + self.text_len + 1]
#         label = torch.LongTensor([self.char_idx[char] for char in label])

#         return sequence, label

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Shakespeare(Dataset):

    def __init__(self, total_text, text_len):
        self.total_text = total_text
        self.chars_set = sorted(set(self.total_text))
        self.char_idx = {char: idx for idx, char in enumerate(self.chars_set)}
        self.idx_char = {idx: char for char, idx in self.char_idx.items()}    
        
        self.text_len = text_len
        self.total_len = len(self.total_text) - self.text_len
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        sequence = self.total_text[idx:idx + self.text_len]
        sequence = [self.char_idx[char] for char in sequence]
        sequence = torch.LongTensor(sequence)
        
        # one-hot encoding
        one_hot_sequence = F.one_hot(sequence, num_classes=len(self.chars_set)).float()
        
        label = self.total_text[idx + 1:idx + self.text_len + 1]
        label = torch.LongTensor([self.char_idx[char] for char in label])

        return one_hot_sequence, label



        
def get_loader(text_file_path, batch_size, text_len, sub_sampling_ratio):
    
    with open(text_file_path, 'rb') as f:
            total_text = ''.join([
                line.strip().lower().decode('ascii', 'ignore')
                for line in f if line.strip()
            ])
            
    chars_set = sorted(set(total_text))
    total_dataset = Shakespeare(total_text, text_len)
    num_samples = len(total_dataset)
    
    train_size = int(sub_sampling_ratio * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, drop_last=True)
    
    return train_loader, val_loader, len(chars_set)
    
if __name__ == '__main__':
    
    text_file_path = '/dshome/ddualab/yuho/deeplearning_course/txt_generation/shakespeare_train.txt'
    
    train_loader, val_loader, chars_set = get_loader(text_file_path, batch_size = 1024, text_len = 30, sub_sampling_ratio = 0.8)
    print(chars_set)
