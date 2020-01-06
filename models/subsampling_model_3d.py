import torch
from torch import nn
from torch.nn import functional as F

from common import args
from models.rec_models.unet_model import UnetModel
from models.rec_models.unet_model import UNet3D
import pytorch_nufft.nufft as nufft
import pytorch_nufft.interp as interp
import data.transforms as transforms
from scipy.spatial import distance_matrix
from tsp_solver.greedy import solve_tsp
import numpy as np


class Subsampling_Layer_3D(nn.Module):
    NUM_MEASUREMENTS_SHOT=500
    def initilaize_trajectory(self,trajectory_learning,initialization):

        num_measurements_shot=self.NUM_MEASUREMENTS_SHOT
        num_shots=self.res*self.depth//self.acceleration_factor

        if initialization=='spiralinspiral':
            x = np.zeros((num_shots, num_measurements_shot, 3))
            rline = np.linspace(0.01, 0.99, num_shots)
            zline = np.linspace(-1, 1, num_measurements_shot)
            for i,val in enumerate(rline):
                x[i,:, 0] = val * zline * np.cos(16 * zline) * self.res / 2
                x[i,:, 1] = val * zline * np.sin(16 * zline) * self.res / 2
                x[i,:, 2] = (1 - val) * zline * self.depth / 2
            x = torch.from_numpy(x).float()
        elif initialization=='sticks':
            num_shots=int(np.sqrt(num_shots))
            x = np.zeros((num_shots**2, num_measurements_shot, 3))
            r = np.linspace(-1, 1, num_measurements_shot)
            theta = np.linspace(0, np.pi, num_shots)
            phi = np.linspace(0, np.pi, num_shots)
            i=0
            for lgn in theta:
                for lat in phi:
                    x[i, :, 0] = r * np.sin(lgn) * np.cos(lat) * self.res / 2
                    x[i, :, 1] = r * np.sin(lgn) * np.sin(lat) * self.res / 2
                    x[i, :, 2] = r * np.cos(lgn) * self.depth / 2
                    i=i+1
            x = torch.from_numpy(x).float()
        elif initialization=='radial':
            # based on matlab spiral
            if num_shots!=312:
                raise NotImplementedError
            x = np.zeros((num_shots, num_measurements_shot, 3))
            r = np.linspace(-1, 1, num_measurements_shot)
            theta = np.load(f'/home/liyon/PILOT/spiral/theta312.npy')[0]
            phi = np.load(f'/home/liyon/PILOT/spiral/phi312.npy')[0]
            for i in range(theta.size):
                x[i, :, 0] = r * np.sin(theta[i]) * np.cos(phi[i]) * self.res / 2
                x[i, :, 1] = r * np.sin(theta[i]) * np.sin(phi[i]) * self.res / 2
                x[i, :, 2] = r * np.cos(theta[i]) * self.depth / 2
            x = torch.from_numpy(x).float()
        elif initialization == 'gaussian':
            #TODO: divide it properly
            x = torch.randn(num_shots, num_measurements_shot, 3)
            x[:,:, 0] = x[:,:, 0] * self.res / 2
            x[:,:, 1] = x[:,:, 1] * self.res / 2
            x[:,:, 2] = x[:,:, 2] * self.depth / 2
        elif initialization == 'uniform':
            x = torch.rand(num_shots, num_measurements_shot, 3)*2-1
            x[:,:, 0] = x[:,:, 0] * self.res / 2
            x[:,:, 1] = x[:,:, 1] * self.res / 2
            x[:,:, 2] = x[:,:, 2] * self.depth / 2
        elif initialization == 'fullkspace':
            x = np.array(np.meshgrid(np.arange(-self.depth / 2, self.depth / 2), np.arange(-self.res / 2, self.res / 2),
                                     np.arange(-self.res / 2, self.res / 2))).T.reshape(num_shots,-1, 3)
            x = torch.from_numpy(x).float()
        elif initialization == '2dradial':
            num_sticks_slice = num_shots//self.depth
            x = np.zeros((num_sticks_slice, num_measurements_shot, 2))
            theta = np.pi / num_sticks_slice
            L = torch.arange(-self.res / 2, self.res / 2, self.res / num_measurements_shot).float()
            for i in range(num_sticks_slice):
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
            self.depthvec=torch.zeros((num_sticks_slice*self.depth,num_measurements_shot,1)).to('cuda')
            start=0
            for d in range(-self.depth//2,self.depth//2):
                self.depthvec[start:start+num_sticks_slice,:,:] = d
                start = start+num_sticks_slice
            x = torch.from_numpy(x).float()
        elif initialization == 'stackofstars':
            num_sticks_slice = num_shots//self.depth
            x = np.zeros((num_sticks_slice, num_measurements_shot, 2))
            theta = np.pi / num_sticks_slice
            L = torch.arange(-self.res / 2, self.res / 2, self.res / num_measurements_shot).float()
            for i in range(num_sticks_slice):
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
            depthvec=torch.zeros((num_sticks_slice*self.depth,num_measurements_shot,1))
            start=0
            for d in range(-self.depth//2,self.depth//2):
                depthvec[start:start+num_sticks_slice,:,:] = d
                start = start+num_sticks_slice
            x = torch.from_numpy(x).float()
            x = x.repeat(self.depth, 1, 1)
            x = torch.cat((x, depthvec), 2)
        else:
            print('Wrong initialization')

        self.x = torch.nn.Parameter(x, requires_grad=trajectory_learning)
        return

    def __init__(self, acceleration_factor, res,depth, trajectory_learning,initialization):
        super().__init__()

        self.acceleration_factor=acceleration_factor
        self.res=res
        self.depth=depth
        self.initialization=initialization
        self.initilaize_trajectory(trajectory_learning, initialization)

    def forward(self, input):
        input = input.squeeze(1).squeeze(1)
        x = self.x
        if self.initialization=='2dradial':
            x=x.repeat(self.depth,1,1)
            x=torch.cat((x,self.depthvec),2)
        x=x.reshape(-1, 3)
        #TODO: Check how this impacts differentiability
        x=torch.max(torch.min(x, torch.tensor([self.depth / 2,self.res / 2,self.res/2]).to('cuda')),
                   torch.tensor([-self.depth / 2,-self.res / 2,-self.res/2]).to('cuda'))

        ksp = interp.bilinear_interpolate_torch_gridsample_3d(input, x)

        output = nufft.nufft_adjoint(ksp, x, input.shape, ndim=3)
        return output.unsqueeze(1)

    def get_trajectory(self):
        x=self.x
        if self.initialization=='2dradial':
            x=self.x.repeat(self.depth,1,1)
            x=torch.cat((x,self.depthvec),2)
        return x

    def __repr__(self):
        return f'Subsampling_Layer'

class Subsampling_Model_3D(nn.Module):
    def __init__(self, in_chans, out_chans, f_maps,acceleration_factor,res,depth, trajectory_learning,initialization):
        super().__init__()

        self.subsampling=Subsampling_Layer_3D(acceleration_factor, res, depth, trajectory_learning,initialization)
        self.reconstruction_model=UNet3D(in_chans,out_chans,True,f_maps=f_maps)


    def forward(self, input):
        input=self.subsampling(input)
        #input,_,__=transforms.normalize_instance(input, eps=1e-11)
        output = self.reconstruction_model(input)
        #output, _, __ = transforms.normalize_instance(output, eps=1e-11)
        return output

    def get_trajectory(self):
        return self.subsampling.get_trajectory()
