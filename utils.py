import torch
import os


def save_checkpoint(state, path, name):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)
        
    torch.save(state, name)
    
    print('Model saved')


def load_checkpoint(path, net, optimizer):
    if not os.path.exists(path):
        raise("File doesn't exist {}".format(path))
        
    state = torch.load(path)
    net.load_state_dict(state['net_state_dict'])
    optimizer.load_state_dict(state['optim_state_dict'])
    
    print('Model loaded')

