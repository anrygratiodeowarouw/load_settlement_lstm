import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import base64

class NSPTExtractor(nn.Module):
    def __init__(self, in_feature=3, hidden_dim=8, out_feature=16):
        super(NSPTExtractor, self).__init__()
        self.cnn1d1 = nn.Conv1d(in_feature, hidden_dim, 3, padding=1, stride=2)
        self.cnn1d2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, stride=2)
        self.cnn1d3 = nn.Conv1d(hidden_dim, out_feature, 3, padding=1, stride=2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.cnn1d1(x))
        x = self.relu(self.cnn1d2(x))
        x = self.relu(self.cnn1d3(x))
        return torch.mean(x, dim=2)
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)

        # Compute input gate
        input_gate = self.sigmoid(self.i2o(combined))

        # Compute output gate
        output_gate = self.sigmoid(self.i2o(combined))

        # Compute cell state
        cell = input_gate * self.tanh(cell) + (1 - input_gate) * self.tanh(self.i2h(combined))

        # Compute hidden state
        hidden = output_gate * self.tanh(cell)

        # Compute output
        output = self.i2o(combined)

        return output, hidden, cell

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Regressor(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, hidden_dim_nspt=8, out_feature_nspt=16, downscale=4, units=1):
        super(Regressor, self).__init__()
        assert len(hidden_size) == units
        self.units = units
        self.nspt_extractor = NSPTExtractor(3, hidden_dim_nspt, out_feature_nspt)
        if downscale == 0:
            self.downscale_input = None
        else:
            self.downscale_input = nn.Linear(feature_size - 1, downscale)

        self.rnn = nn.ModuleList()
        if units > 1:
            for i in range(units):
                if i == 0:
                    self.rnn.append(LSTM(downscale + 1 + out_feature_nspt if downscale != 0 else feature_size + out_feature_nspt, hidden_size[i], hidden_size[i]))
                elif i == units - 1:
                    self.rnn.append(LSTM(hidden_size[i - 1], hidden_size[i], output_size))
                else:
                    self.rnn.append(LSTM(hidden_size[i - 1], hidden_size[i], hidden_size[i]))
        else:
            self.rnn.append(LSTM(downscale + 1 + out_feature_nspt if downscale != 0 else feature_size + out_feature_nspt, hidden_size[0], output_size))

    def forward(self, input, nspt, print_vector=False):
        batch_size = input.shape[0]
        if self.downscale_input is not None:
            feature1 = input[:,:,:-1]
            feature2 = input[:,:,-1].unsqueeze(2)
            feature1 = self.downscale_input(feature1)
            input = torch.cat((feature1, feature2), dim=2) #concatenate feature1 downscale and feature2 ei

        nspt = self.nspt_extractor(nspt)
        nspt = nspt.unsqueeze(1).repeat(1, input.shape[1], 1)
        input = torch.cat((input, nspt), dim=2)

        hidden = []
        for i in range(len(self.rnn)):
            hidden.append(self.rnn[i].init_hidden(batch_size).to(device))
        cell = []
        for i in range(len(self.rnn)):
            cell.append(self.rnn[i].init_hidden(batch_size).to(device))

        output = torch.zeros(input.shape[0], input.shape[1], 1)
        for i in range(input.shape[1]): #iterate trough sequence
            if self.units > 1:
                for j in range(len(self.rnn)):
                    if j == 0:
                        out, hidden[j], cell[j] = self.rnn[j](input[:,i,:], hidden[j], cell[j])
                    elif j == len(self.rnn)-1:
                        output[:,i,:], hidden[j], cell[j] = self.rnn[j](out, hidden[j], cell[j])
                    else:
                        out, hidden[j], cell[j] = self.rnn[j](out, hidden[j], cell[j])
            else:
                output[:,i,:], hidden[0], cell[0] = self.rnn[0](input[:,i,:], hidden[0], cell[0])

        if print_vector:
            print("input vector: ", input)
            for i in range(len(self.rnn)):
                print("hidden vector ", i, ": ", hidden[i])
                print("cell vector ", i, ": ", cell[i])
            print("output vector: ", output)
        return output
    
scaler_x = pickle.load(open("scaler_x_14.05.pkl", "rb"))
scaler_nspt = pickle.load(open("scaler_nspt_14.05.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y_14.05.pkl", "rb"))
le = pickle.load(open("scaler_label_encorder_14.05.pkl", "rb"))

def generate_ei(n_ei):
    res = []
    current_value = 0
    for i in range(n_ei):
        current_value += 0.05*i
        res.append(current_value)
    return res

def predict_qi(d, l, col, n_ei, nspt_df,model):
    ei = generate_ei(n_ei)
    nspt = nspt_df.values

    X = [[d, l, col, ei[i]] for i in range(len(ei))]
    
    nspt[:,2] = le.transform(nspt[:,2]).tolist()
    X_transformed = scaler_x.transform(X)
    nspt_transformed = scaler_nspt.transform(np.array(nspt).reshape(-1,3))
    x = torch.tensor(X_transformed).float().to(device).unsqueeze(0)
    nspt = torch.tensor(nspt_transformed.reshape(3, -1)).float().to(device).unsqueeze(0)
    y_pred = model(x, nspt).to(device)
    y_pred = scaler_y.inverse_transform(y_pred.cpu().detach().numpy().reshape(-1,1))
    y_pred[0,0] = 0
    return y_pred

def plot_prediction(n_ei, prediction):
    ei = generate_ei(n_ei)

    fig, ax = plt.subplots()
    ax.plot(ei, prediction, label='pred')
    ax.legend()

    return fig
