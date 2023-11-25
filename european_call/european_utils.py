import numpy as np
import torch

class EuropeanOptionData():
    def __init__(self,t_range,S_range,K,r, sigma):
        self.t_range = t_range
        self.S_range = S_range
        self.K = K
        self.r = r
        self.sigma = sigma

    def _gs(self,x):
        return np.fmax(x-self.K, 0)    

    def get_diff_data(self,n):
        X = np.concatenate([np.random.uniform(*self.t_range, (n, 1)),
                        np.random.uniform(*self.S_range, (n, 1))], axis=1)
        y = np.zeros((n, 1)) #price
        return X, y
    
    def get_diff_data_tensor(self,N_sample,mul=4):
        X1, y1 = self.get_diff_data(mul*N_sample)
        X1 = torch.from_numpy(X1).float().requires_grad_()
        y1 = torch.from_numpy(y1).float()
        return X1,y1
    
    def get_ivp_data(self,n):
        X = np.concatenate([np.ones((n, 1)), #all at expiry time
                    np.random.uniform(*self.S_range, (n, 1))], axis=1)
        y = self._gs(X[:, 1]).reshape(-1, 1)
        return X, y
    
    def get_ivp_data_tensor(self,N_sample):
        ivp_x, ivp_y = self.get_ivp_data(N_sample)
        ivp_x_tensor = torch.from_numpy(ivp_x).float()
        ivp_y_tensor = torch.from_numpy(ivp_y).float()
        return ivp_x_tensor,ivp_y_tensor
        
    
    def get_bvp_data(self,n):
        T = self.t_range[-1]
        X1 = np.concatenate([np.random.uniform(*self.t_range, (n, 1)),
                        self.S_range[0] * np.ones((n, 1))], axis=1)
        y1 = np.zeros((n, 1))
        X2 = np.concatenate([np.random.uniform(*self.t_range, (n, 1)),
                        self.S_range[-1] * np.ones((n, 1))], axis=1)
        y2 = (self.S_range[-1] - self.K*np.exp(-self.r*(T-X2[:, 0].reshape(-1)))).reshape(-1, 1)
        return X1, y1, X2, y2
    
    def get_bvp_data_tensor(self,N_sample):
        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = self.get_bvp_data(N_sample)
        bvp_x1_tensor = torch.from_numpy(bvp_x1).float()
        bvp_y1_tensor = torch.from_numpy(bvp_y1).float()
        bvp_x2_tensor = torch.from_numpy(bvp_x2).float()
        bvp_y2_tensor = torch.from_numpy(bvp_y2).float()
        return bvp_x1_tensor,bvp_y1_tensor,bvp_x2_tensor,bvp_y2_tensor
    
    def get_analytical_soln(self, S, t):
        t2m = t  # Time to maturity (assumed in years)
        d1 = (torch.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * t2m) / (self.sigma * torch.sqrt(t2m))
        d2 = d1 - self.sigma * torch.sqrt(t2m)
        # Normal cumulative distribution function (CDF)
        N0 = lambda value: 0.5 * (1 + torch.erf(value / (2**0.5)))
        Nd1 = N0(d1)
        Nd2 = N0(d2)
        # Calculate the option price
        C = S * Nd1 - self.K * Nd2 * torch.exp(-self.r * t2m)
        return C