# References

## Full list of options from the script

```
Grid PyTorch Lightning Optuna example.

optional arguments:
  -h, --help            show this help message and exit
  --pruning {0,1}, -p {0,1}
                        Activate Optuna pruning feature. `MedianPruner` stops unpromising trials at the early stages of training. (default: 0)
  --datadir DATADIR     FashionMNIST directory (default: /Users/robertlee/github/grid-optuna)
  --batchsize BATCHSIZE
                        Batchsize (default: 128)
  --epochs EPOCHS       Max epochs (default: 10)
  --timeout TIMEOUT     Max seconds to run (default: 60)
  --gpus GPUS           Number of GPUs to use (default: 0)
```

## Creation of `requirements.txt`

`requirements.txt` creating using `grid sync-env`

```bash
touch requirements.txt
grid sync-env
git add requirements.txt
git commit -m "requirements.txt synced with current environment"
```

## Changes to the [original](https://raw.githubusercontent.com/optuna/optuna-examples/main/pytorch/pytorch_lightning_simple.py) code

Convert hard coded variables to be command line arguments
  
```bash
curl -O https://raw.githubusercontent.com/optuna/optuna-examples/main/pytorch/pytorch_lightning_simple.py
chmod a+x pytorch_lightning_simple.py

diff pytorch_lightning_simple.py ~/github/optuna-examples/pytorch/pytorch_lightning_simple.py > patchfile.patch
36c36
< BATCHSIZE = 128 # make this parameter
---
> BATCHSIZE = 128
38,39c38,39
< EPOCHS = 10 # make this parameter
< DIR = os.getcwd() # make this parameter
---
> EPOCHS = 10
> DIR = os.getcwd()
129c129
<     datamodule = FashionMNISTDataModule(data_dir=args.datadir, batch_size=args.batchsize)
---
>     datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
135,136c135,136
<         max_epochs=args.epochs,
<         gpus=args.gpus,
---
>         max_epochs=EPOCHS,
>         gpus=1 if torch.cuda.is_available() else None,
139c139
<     hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims, epoch=args.epochs, batchsize=args.batchsize)
---
>     hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
147,148c147
<     parser = argparse.ArgumentParser(description="Grid PyTorch Lightning Optuna example.",
<         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
---
>     parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
152,153c151,152
<         default=0, type=int, choices=[0,1],
<         help="Activate Optuna pruning feature. `MedianPruner` stops unpromising "
---
>         action="store_true",
>         help="Activate the pruning feature. `MedianPruner` stops unpromising "
156,160d154
<     parser.add_argument('--datadir', default=f'{os.getcwd()}', type=str, help="FashionMNIST directory")
<     parser.add_argument('--batchsize', default=BATCHSIZE, type=int, help="Batchsize")
<     parser.add_argument('--epochs', default=EPOCHS, type=int, help="Max epochs")
<     parser.add_argument('--timeout', default=60, type=int, help="Max seconds to run")
<     parser.add_argument('--gpus', default=0, type=int, help="Number of GPUs to use")
168c162
<     study.optimize(objective, n_trials=100, timeout=args.timeout)
---
>     study.optimize(objective, n_trials=100, timeout=600)
```