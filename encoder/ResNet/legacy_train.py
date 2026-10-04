import json
import time
import torch
import torch.nn as nn
from pathlib import Path

from network import *
from dataloader import *
from torch.utils.data import random_split

data_length = 1000
backbone = 'resnet18'
weights = {}
timestr = time.strftime(r"%Y%m%d-%H%M%S", time.localtime())

class ReconstructionModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.reconstruct(image)

def result_dir():
    return (Path(__file__).resolve().parents[2] / 'encoder_results' / 'marker_rgb' /
            'ResNet' / 'S' / timestr)

def log(msg):
    save_path = result_dir() / 'log.log'
    with open(save_path, 'a') as f:
        f.write(msg + '\n')

def train(
    train_loader:DataLoader,
    valid_loader:DataLoader,
    epoches=5,
    lr=1e-4,
    loss_weights:dict={'marked_rgb': 1.0},
    device=None,
    multi_gpu=False,
):
    global backbone, timestr, weights
    save_root = result_dir()
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / 'config.json').write_text(json.dumps({
        'format_version': 0, 'checkpoint_format': 'legacy_raw_resnet18',
        'method': 'ResNet', 'size': 'S', 'input_mode': 'marker_rgb',
        'samples': data_length, 'output_dir': str(save_root),
    }, indent=2) + '\n')
    log(json.dumps({
        'status': 'config',
        'backbone': backbone,
        'epoches': epoches,
        'lr': lr,
        'loss_weights': loss_weights,
    }, ensure_ascii=False))

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    supervise = list(loss_weights.keys())
    model = Tactile(backbone=backbone, supervise=supervise).to(device)
    runner = ReconstructionModel(model)
    if multi_gpu and device.type == 'cuda' and torch.cuda.device_count() > 1:
        runner = nn.DataParallel(runner)
    optimizer = torch.optim.Adam(runner.parameters(), lr=lr)
    
    train_losses = []
    valid_losses = []
    best_loss = 1e9
    
    for epoch in range(epoches):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        pbar.set_description(f'Epoch {epoch+1:4d}/{epoches:4d}, Training')
        for idx, d in pbar:
            x:torch.Tensor = d['marked_rgb'].to(device)
            y = {s: d[s].to(device) for s in supervise}
            outputs = runner(x)

            loss, loss_dict = model.loss(outputs, y, weights=loss_weights)
            train_loss += loss_dict['total']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss_dict['total']})
            if idx % 20 == 0:
                log(json.dumps({
                    'status': 'train',
                    'epoch': epoch + 1,
                    'batch': idx + 1,
                    'loss': loss_dict['total'],
                    'losses': loss_dict
                }, ensure_ascii=False))
            
        train_losses.append(train_loss / len(train_loader))

        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            pbar.set_description(f'Epoch {epoch+1:4d}/{epoches:4d}, Validating')
            for idx, d in pbar:
                x:torch.Tensor = d['marked_rgb'].to(device)
                y = {s: d[s].to(device) for s in supervise}
                outputs = runner(x)

                loss, loss_dict = model.loss(outputs, y, weights=loss_weights)
                valid_loss += loss_dict['total']
                
                pbar.set_postfix({'loss': loss_dict['total']})
                if idx % 20 == 0:
                    log(json.dumps({
                        'status': 'eval',
                        'epoch': epoch + 1,
                        'batch': idx + 1,
                        'loss': loss_dict['total'],
                        'losses': loss_dict
                    }, ensure_ascii=False))
        
        torch.save(model.state_dict(), str(save_root / f'ep{epoch}.pth'))
        valid_losses.append(valid_loss / len(valid_loader))
        if valid_losses[-1] < best_loss:
            best_loss = valid_losses[-1]
            torch.save(model.state_dict(), str(save_root / 'best.pth'))
            print(f'  Best model saved with recon loss {best_loss:.6f}')

def main(args):
    global data_length
    prism_names = ['CircleShell', 'Cross', 'Cubehole', 'Cuboid', 'Cylinder', 'Doubleslope', 'Hemisphere', 'Line', 'Pacman', 'S', 'Sphere', 'Star', 'Tetrahedron', 'Torus']
    hdf5_paths = []
    for name in prism_names:
        l = sorted((args.data_root / name / 'hdf5').glob('*.hdf5'))
        hdf5_paths.extend(l)
    print(f'Found {len(hdf5_paths)} hdf5 files.')
    if not hdf5_paths:
        raise FileNotFoundError(f'No HDF5 files found under {args.data_root}')
    data = HDF5Dataset(hdf5_paths)
    if data_length < 2 or data_length > len(data):
        raise ValueError(f'data_num must be between 2 and {len(data)}, got {data_length}')
    data._data_metadata = list(np.array(data._data_metadata)[
        np.random.choice(len(data._data_metadata), data_length, replace=False)])
    print(f'Dataset length: {len(data)}')
    
    batch_size = args.batch_size
    pin_memory = args.device == 'cuda' or (args.device is None and torch.cuda.is_available())
    valid_size = max(1, int(len(data) * 0.2))
    generator = torch.Generator().manual_seed(42)
    train_size = len(data) - valid_size
    train_data, valid_data = random_split(
        data, [train_size, valid_size], generator=generator)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=args.workers,
        persistent_workers=args.workers > 0, worker_init_fn=worker_init_fn, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, worker_init_fn=worker_init_fn, pin_memory=pin_memory,
    )
    train(train_loader, valid_loader, epoches=args.epochs, lr=1e-3, loss_weights=weights,
          device=args.device, multi_gpu=args.multi_gpu)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('config', type=str, nargs='?', default='marker_rgb', choices=['marker_rgb'])
    parser.add_argument('data_num', type=int, nargs='?', default=1000)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[2] / 'resources/data/contact')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--multi-gpu', action='store_true')
    args = parser.parse_args()
    
    backbone = 'resnet18'
    data_length = args.data_num
    weights = {'rgb': 1.0, 'marker': 1.0}
    main(args)
