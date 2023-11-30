import numpy as np
import torch
import matplotlib.pyplot as plt

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
        
    
    def get_bvp_data(self,n,r1=1,r2=1):
        T = self.t_range[-1]
        X1 = np.concatenate([np.random.uniform(*self.t_range, (int(n*r1), 1)),
                        self.S_range[0] * np.ones((n, 1))], axis=1)
        y1 = np.zeros((n, 1))
        X2 = np.concatenate([np.random.uniform(*self.t_range, (int(r2*n), 1)),
                        self.S_range[-1] * np.ones((n, 1))], axis=1)
        y2 = (self.S_range[-1] - self.K*np.exp(-self.r*(T-X2[:, 0].reshape(-1)))).reshape(-1, 1)
        return X1, y1, X2, y2
    
    def get_bvp_data_tensor(self,N_sample,r1=1,r2=1):
        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = self.get_bvp_data(N_sample,r1,r2)
        bvp_x1_tensor = torch.from_numpy(bvp_x1).float()
        bvp_y1_tensor = torch.from_numpy(bvp_y1).float()
        bvp_x2_tensor = torch.from_numpy(bvp_x2).float()
        bvp_y2_tensor = torch.from_numpy(bvp_y2).float()
        return bvp_x1_tensor,bvp_y1_tensor,bvp_x2_tensor,bvp_y2_tensor
    
    def normalize(self,x):
        min_values = torch.tensor([self.t_range[0],self.S_range[0]])
        max_values = torch.tensor([self.t_range[1],self.S_range[1]])
        normalized_tensor = (2 * (x - min_values) / (max_values - min_values)) - 1
        return normalized_tensor
    
    def get_analytical_soln(self, S, t):
        T = self.t_range[-1]
        t2m = T-t  # Time to maturity (assumed in years)
        d1 = (torch.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * t2m) / (self.sigma * torch.sqrt(t2m))
        d2 = d1 - self.sigma * torch.sqrt(t2m)
        # Normal cumulative distribution function (CDF)
        N0 = lambda value: 0.5 * (1 + torch.erf(value / (2**0.5)))
        Nd1 = N0(d1)
        Nd2 = N0(d2)
        # Calculate the option price
        C = S * Nd1 - self.K * Nd2 * torch.exp(-self.r * t2m)
        return C
    

def plot_solution(model,euro_call_data,i, experiment_dir, close=True):
  s = np.linspace(euro_call_data.S_range[0], euro_call_data.S_range[1], 50)
  t = np.linspace(euro_call_data.t_range[0], euro_call_data.t_range[1], 50)
  s_grid, t_grid = np.meshgrid(s, t)
  s_flat = s_grid.flatten()
  t_flat = t_grid.flatten()
  # Create a 2D tensor from the flattened arrays
  X_test = torch.tensor(np.column_stack((t_flat, s_flat)), dtype=torch.float)
  y_analytical_test = euro_call_data.get_analytical_soln(X_test[:,1],X_test[:,0])
  model.eval();
  with torch.no_grad():
    y_pinn_test = model(X_test)
  # Create the 3D plot
  fig = plt.figure(figsize=(14,7))
  ax = fig.add_subplot(121, projection='3d')
  ax.plot_surface(s_grid, t_grid, y_analytical_test.detach().numpy().reshape(s_grid.shape), cmap = "viridis")
  ax.set_title("Analytical Soln")
  ax.set_xlabel("Spot Price")
  ax.set_ylabel("Current time")
  ax.set_zlabel("Call price")
  ax.view_init(elev=20, azim=-120)
  ax = fig.add_subplot(122, projection='3d')
  ax.plot_surface(s_grid, t_grid, y_pinn_test.detach().numpy().reshape(s_grid.shape), cmap = "viridis")
  ax.set_title("PINN prediction")
  ax.set_xlabel("Spot Price")
  ax.set_ylabel("Current time")
  ax.set_zlabel("Call price")
  ax.view_init(elev=20, azim=-120)
  if close:
    plt.savefig(experiment_dir+f"/true_vs_pred_{i}.jpg")
    plt.close()
  else:
    plt.savefig(experiment_dir+f"/true_vs_pred_{i}.jpg")
  model.train();    