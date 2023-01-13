from __future__ import annotations
import numpy as np
import torch
import h5py
import argparse
from ..models.sampler import SamplingModel
from geomloss import SamplesLoss
import re
from torch.utils.data import DataLoader, Dataset, random_split
from typing import NamedTuple, Optional, Callable
from random import choices

def draw_samples(A, n, dtype=torch.FloatTensor):
    xg, yg = np.meshgrid(
        np.linspace(0, 1, A.shape[1]),
        np.linspace(0, 1, A.shape[0]),
        indexing="xy",
    )
    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() 
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / A.shape[0]) * np.random.standard_normal(dots.shape)
    return torch.from_numpy(dots).type(dtype)

class SamplerArguments(NamedTuple):
    cmd: str
    device: int
    embedding: str
    iter: Optional[int]
    lr: Optional[float]
    max_data: Optional[int]
    save_at_iter: Optional[int]
    output: str
    checkpoint: Optional[str]
    func: Callable[[SamplerArguments], None]



def add_args(parser):
    parser.add_argument("--embedding", help = "Embedding file")
    parser.add_argument("--checkpoint", default = None, help = "Checkpoint for training sample")
    parser.add_argument("--iter", default = 10, type = int, help = "Number of iterations")
    parser.add_argument("--output", help = "output folder")
    parser.add_argument("--save-at-iter", default = 1, type = int, help = "Save the model at this iteration")
    parser.add_argument("--lr", default = 1, type = float, help = "Learning rate")
    parser.add_argument("--device", default = 1, type = int, help = "CUDA device")
    parser.add_argument("--max-data", default = 1000, type = int, help = "CUDA device")
    return parser


class CmapData(Dataset):
    def __init__(self, h5locs, no_samples = 100, max_row = 400, max_col = 400, 
                 max_data = -1, preprocess = None):
        super(CmapData, self).__init__()
        self.h5files = [h5py.File(h5loc, "r") for h5loc in h5locs]
        self.no_samples = no_samples
        self.max_row = max_row
        self.max_col = max_col
        self.preprocess = preprocess
        
        self.counts = [0]
        self.h5samples = []
        for f in self.h5files:
            self.h5samples += list(f.keys())
            self.counts.append(len(self.h5samples))
        
        self.max_data = len(self.h5samples) if max_data < 0 else max_data
        
    
    def get_loc(self, id):
        h5id = 0
        for i in self.counts[1:]:
            if id >= i:
                h5id += 1
            else:
                return self.h5files[h5id]
                
    def __len__(self):
        return self.max_data
    
    def __getitem__(self, id):
        assert id < self.max_data
        
        key = self.h5samples[id]
        h5sample = self.get_loc(id)
        A = h5sample.get(key)[()]
        
        if self.preprocess is not None:
            A = self.preprocess(A)
        m, n = A.shape
        
        pad_y = self.max_row - m
        pad_x = self.max_col - n
        
        A = np.pad(A, ((0, pad_y), (0, pad_x)))

        X = draw_samples(A, self.no_samples, dtype = torch.float32)
        A = torch.tensor(A, dtype = torch.float32).unsqueeze(0)
        return A, X
    


def main(args):
    """
    Sampler Training module
    """
    device = "cpu"
    regex_comp   = re.compile(r"_(\d*).sav")
    current_iter = 0
    if torch.cuda.is_available():
        device = torch.device(args.device)
        
    if args.checkpoint is not None:
        model = torch.load(args.checkpoint, map_location = device)
        current_iter = int(regex_comp.search(args.checkpoint).group(1))
    else:
        model = SamplingModel().to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    lossf     = SamplesLoss("sinkhorn", p = 2, blur = 0.1)
    
    train_dset   = CmapData([args.embedding], 100, preprocess = None, max_data = args.max_data)
    train_loader = DataLoader(train_dset, batch_size = 1, shuffle = True)
    ofile     = open(f"{args.output}/logs.txt", "a")
    
    for i in range(current_iter + 1, args.iter):
        running_loss = 0
        for e, data in enumerate(train_loader):
            A, Y = data
            A = A.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            Y_pred = torch.transpose(model(A), 1, 2).squeeze(0).squeeze(0)
            loss = lossf(Y.squeeze(0), Y_pred)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if device.type != "cpu":
                A = A.to("cpu")
                Y = Y.to("cpu")
                loss = loss.to("cpu")
                Y_pred = Y_pred.to("cpu")
        running_loss /= (e+1)
        print(f"Running Interation {i+1}, Loss: {running_loss}")
        ofile.write(f"Running Interation {i+1}, Loss: {running_loss}\n")
        ofile.flush()
        if (i+1) % args.save_at_iter == 0:
            torch.save(model, f"{args.output}/iter_{i}.sav")
    ofile.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    add_args(parser)
    main(parser.parse_args())