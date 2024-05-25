# import some packages you need here
import dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CharLSTM, CharRNN

from dataset import Shakespeare

def generate(model, seed_characters, temperature=1, char_idx = None, idx_char = None, gen_len = None):

    model.eval()
    batch_size = 1

    char_li = [char_idx[character] for character in seed_characters] 
    hidden_init = model.init_hidden(batch_size)

    X = np.zeros((1, gen_len+1, len(char_idx)))
    X = torch.FloatTensor(X)
    
    # one-hot
    for i, idx in enumerate(char_li) :
        X[:, i, idx] = 1   
    
    total = []
    total += seed_characters
    
    for num in range(gen_len - len(char_li)):
        
        num += len(char_li)
        
        output, _ = model(X[:,:num+1,:], hidden_init)
        
        out_distribution = F.softmax(output/temperature, dim=-1)[:,-1,:]
        pred_ind = torch.multinomial(out_distribution, num_samples=1).item()
        
        # next word : one-hot
        X[:,num+1, pred_ind] = 1
        # next word : append 
        total.append(idx_char[pred_ind])
        
    return "".join(total)
        
    
if __name__ == '__main__':
    text_file_path = '/dshome/ddualab/yuho/deeplearning_course/txt_generation/shakespeare_train.txt'
    
    text_len = 30
    
    model_name = "LSTM"
    hidden_size = 64
    num_layers = 2
    
    dataset_ = Shakespeare(text_file_path, text_len)    
    
    if model_name == 'RNN':
        model = CharRNN(36, hidden_size, 36, num_layers=num_layers)
    elif model_name == 'LSTM':
        model = CharLSTM(36, hidden_size, 36, num_layers=num_layers)
    
    ckpt = torch.load("/dshome/ddualab/yuho/deeplearning_course/txt_generation/best_LSTM_model.pt")
    model.load_state_dict(ckpt)

    
    with open(text_file_path, 'rb') as f:
            total_text = ''.join([
                line.strip().lower().decode('ascii', 'ignore')
                for line in f if line.strip()
            ])
            
    dataset = Shakespeare(total_text, text_len)   
    seed_characters = 'you'
    temperature = 0.3
    result = generate(model, seed_characters, temperature, dataset.char_idx, dataset.idx_char, gen_len = 200)      
    
    print(result)