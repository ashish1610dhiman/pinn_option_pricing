import torch.nn as nn
import torch
# import torch.nn.init as init

class EuropeanCall_gated1(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(EuropeanCall_gated1, self).__init__()
        self.N_HIDDEN = N_HIDDEN
        self.activation1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation2 = nn.Tanh()
        self.fcs1 = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            self.activation2)
        self.fcs2 = nn.Sequential(
            nn.Linear(N_INPUT, N_OUTPUT),
            self.activation2)
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                self.activation2
            ) for _ in range(N_LAYERS)])
        self.fce = nn.Sequential(*[nn.Linear(N_HIDDEN, N_OUTPUT)])
        self.w1_layer = nn.Sequential(*[nn.Linear(N_HIDDEN + N_INPUT, N_OUTPUT),self.activation2])
        self.w2_layer = nn.Sequential(*[nn.Linear(N_HIDDEN + N_INPUT, N_OUTPUT),self.activation2])

    def forward(self, x):
        # Apply the first layer
        I1 = self.fcs1(x)
        H = I1
        # Apply hidden layers with residual connections
        for layer in self.fch:
            H = layer(H) + H
        # Apply the final layer
        yx = self.fcs2(x) #1D
        yh =  self.fce(H)#1D
        h_x = torch.cat([H,x],axis=1)
        # print (h_x.shape)
        wx = self.w1_layer(h_x)
        wh = self.w2_layer(h_x)
        y_net = wx*yx + wh*yh
        return y_net
    

class EuropeanCall_gated2(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(EuropeanCall_gated2, self).__init__()
        self.N_HIDDEN = N_HIDDEN
        self.activation1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation2 = nn.Tanh()
        self.fcs1 = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            self.activation2)
        self.fcs2 = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            self.activation2)
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                self.activation2
            ) for _ in range(N_LAYERS)])
        self.fce1 = nn.Sequential(*[nn.Linear(N_HIDDEN, N_OUTPUT)])
        self.fce2 = nn.Sequential(*[nn.Linear(N_HIDDEN, N_OUTPUT)])
        self.w1_layer = nn.Sequential(*[nn.Linear(N_HIDDEN + N_HIDDEN, N_OUTPUT),self.activation2])
        self.w2_layer = nn.Sequential(*[nn.Linear(N_HIDDEN + N_HIDDEN, N_OUTPUT),self.activation2])
        # init.xavier_uniform_(self.fcs1.weight)
        # init.xavier_uniform_(self.fcs2.weight)
        # init.xavier_uniform_(self.fch.weight)

    def forward(self, x):
        # Apply the first layer
        I1 = self.fcs1(x)
        H1 = I1
        # Apply hidden layers with residual connections
        for layer in self.fch:
            H1 = layer(H1) + H1
        # Apply the final layer
        H2 = self.fcs2(x) #1
        y1 =  self.fce1(H1)#1D
        y2 =  self.fce2(H2)#1D
        H_cat = torch.cat([H1,H2],axis=1)
        # print (h_x.shape)
        w1 = self.w1_layer(H_cat)
        w2 = self.w2_layer(H_cat)
        y_net = w1*y1 + w2*y2
        return y_net

class EuropeanCall_gated3(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(EuropeanCall_gated3, self).__init__()
        self.N_HIDDEN = N_HIDDEN
        self.activation1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation2 = nn.Tanh()
        self.fcs1 = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            self.activation2)
        self.fcs2 = nn.Sequential(
            nn.Linear(N_INPUT, N_OUTPUT),
            self.activation2)
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                self.activation2
            ) for _ in range(N_LAYERS)])
        self.fce = nn.Sequential(*[nn.Linear(N_HIDDEN, N_OUTPUT)])
        self.w1_layer = nn.Sequential(*[nn.Linear(N_HIDDEN + N_INPUT, N_OUTPUT),self.activation2])
        # self.w2_layer = nn.Sequential(*[nn.Linear(N_HIDDEN + N_INPUT, N_OUTPUT),self.activation2])

    def forward(self, x):
        # Apply the first layer
        I1 = self.fcs1(x)
        H = I1
        # Apply hidden layers with residual connections
        for layer in self.fch:
            H = layer(H) + H
        # Apply the final layer
        yx = self.fcs2(x) #1D
        yh =  self.fce(H)#1D
        h_x = torch.cat([H,x],axis=1)
        # print (h_x.shape)
        wh = self.w1_layer(h_x)
        y_net = yx + wh*yh
        return y_net
    

class DGMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, n_layers=3, output_dim=1):
    super(DGMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.n = n_layers
    self.sig_act = nn.Tanh()
    self.Sw = nn.Linear(self.input_dim, self.hidden_dim)
    self.Uz = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsz = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.Ug = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsg = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.Ur = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsr = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.Uh = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsh = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.Wf = nn.Linear(hidden_dim, output_dim)
    

  def forward(self, x):
    S1 = self.Sw(x)
    for i in range(self.n):
      if i==0:
        S = S1
      else:
        S = self.sig_act(out)
      Z = self.sig_act(self.Uz(x) + self.Wsz(S))
      G = self.sig_act(self.Ug(x) + self.Wsg(S1))
      R = self.sig_act(self.Ur(x) + self.Wsr(S))
      H = self.sig_act(self.Uh(x) + self.Wsh(S*R))
      out = (1-G)*H + Z*S
    out = self.Wf(out)
    return out    
