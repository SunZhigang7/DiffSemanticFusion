from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import NuscenesDatasetInMemory
from predictors import QCNet
from transforms import NuscenesTargetBuilder

if __name__ == '__main__':

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No available GPU")

    pl.seed_everything(2025, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=False)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    # QCNet.add_model_specific_args(parser)
    # args = parser.parse_args()
    # model = QCNet(**vars(args))

    # checkpoint = torch.load(args.ckpt_path)
    # model.load_state_dict(checkpoint['state_dict'])

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)

    val_dataset = {
        'nuscenes': NuscenesDatasetInMemory,
    }[model.dataset](root=args.root, split='val',
                     transform=NuscenesTargetBuilder(model.num_historical_steps, model.num_future_steps))

    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    trainer.validate(model, dataloader)
