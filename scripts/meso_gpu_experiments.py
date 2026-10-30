"""Reusable GPU forward execution for a fixed-size inference batch."""
import torch

from toy_meso.common import autocast


class CUDAForward:
    """Replay model operations while copying each call's inputs into fixed buffers."""
    def __init__(self, model, shape, device):
        self.model = model
        self.x = torch.zeros(shape, dtype=torch.float32, device=device)
        self.condition = torch.zeros_like(self.x)
        self.sigma = torch.ones(shape[0], dtype=torch.float32, device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.no_grad(), autocast(device):
            for _ in range(3):
                model(self.x, self.sigma, self.condition)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad(), autocast(device):
            self.output = model(self.x, self.sigma, self.condition)

    def __call__(self, x, sigma, condition):
        self.x.copy_(x)
        self.sigma.copy_(sigma)
        self.condition.copy_(condition)
        self.graph.replay()
        return self.output
