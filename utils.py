import tqdm
from IPython import display
import torchvision
from torchvision.transforms import functional as TF
import torch

def train(model, data, optim, loss_fn, ema = .95):
    model.train()
    total_loss = None
    bar = tqdm.tqdm(data)
    for x, y in bar:
        optim.zero_grad()
        yp = model(x)
        loss = loss_fn(yp, y)
        if total_loss is None:
            total_loss = loss.item()
        else:
            total_loss = ema * total_loss + (1-ema) * loss.item()
        bar.set_description(f'Loss: {total_loss:g}')
        loss.backward()
        optim.step()

def show_image(images):
    grid = torchvision.utils.make_grid(images, int(len(images)**.5)).cpu()
    grid = TF.to_pil_image(grid.clamp(0, 1))
    display.display(grid)

def save(model:torch.nn.Module, opt:torch.optim.Optimizer, name="model.pth"):
    torch.save((model.state_dict(), opt.state_dict()), name)

def load(model:torch.nn.Module, opt:torch.optim.Optimizer, name="model.pth"):
    sd = torch.load(name)
    model.load_state_dict(sd[0])
    opt.load_state_dict(sd[1])
