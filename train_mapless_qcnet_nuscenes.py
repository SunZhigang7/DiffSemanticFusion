from argparse import ArgumentParser
import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader
from datasets import NuscenesDatasetInMemory
from transforms import NuscenesTargetBuilder
from predictors import QCNet

if __name__ == '__main__':

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No available GPU")
        
    pl.seed_everything(2025, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    # save RAM
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=False)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = QCNet(**vars(args))
  
    train_dataset = {
        'nuscenes': NuscenesDatasetInMemory,
    }[model.dataset](root=args.root, split='train',
                     transform=NuscenesTargetBuilder(model.num_historical_steps, model.num_future_steps))
    
    val_dataset = {
        'nuscenes': NuscenesDatasetInMemory,
    }[model.dataset](root=args.root, split='val',
                     transform=NuscenesTargetBuilder(model.num_historical_steps, model.num_future_steps))

    model_checkpoint = ModelCheckpoint(monitor='val_minADE', save_top_k=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
    #                     strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
    #                     callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, val_check_interval=0.5)
    
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                    strategy="ddp_spawn", 
                    callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, check_val_every_n_epoch=1)

    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    