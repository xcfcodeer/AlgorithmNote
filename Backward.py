# conda activate SoftwareDevelop

import logging
import numpy as np
import copy
import torch
import torch.nn as nn

def log_testing(filename='fem.txt'):
    logging.basicConfig(filename=filename,filemode='a',level=logging.INFO,format='%(asctime)s-%(name)s- %(levelname)s-%(message)s-%(funcName)s')

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 第一层：输入维度1，输出维度3
        self.layer1 = nn.Linear(1, 3)
        # 第二层（隐藏层）：输入维度3，输出维度1
        self.layer2 = nn.Linear(3, 1)
        with torch.no_grad():  # 在这个上下文中，不需要计算梯度
            self.layer1.weight.fill_(1.0)  # 权重设置为1
            self.layer1.bias.fill_(1.0)
            self.layer2.weight.fill_(1.0) 
            self.layer2.bias.fill_(1.0)


    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def MSE_loss_test(y_t,y):
    return 1/2 *(y_t-y)**2

if __name__ =='__main__':
    log_testing(filename=r'F:\CodeFarmer\game_program\log_file\program_develop.txt')
    logging.info(f"研究pytorch框架中的反向传播算法。")
    #-- 研究反向传播算法中涉及的数据结构。
    model = SimpleNN()
    in_ = torch.tensor([1.0])
    out_= model(in_)
    logging.info(f"output result: {out_}")
    # model.parameters是一个generators
    for param in model.parameters():
        logging.info(f"model parameters: {param}")
    y_label= torch.tensor([2]).float()
    loss_fn= torch.nn.MSELoss() # 没有1/2 
    loss_value = loss_fn(y_label,out_)
    logging.info(f"loss result: {loss_value}")
    learning_rate= 0.01
    optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)
    # update 
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    for param in model.parameters():
        logging.info(f"model parameters: {param}")

