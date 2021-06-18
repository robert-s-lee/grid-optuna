[Grid.ai](https://www.grid.ai) can seamlessly train 100s of machine learning models on the cloud from your laptop, with zero code change.
In this example, we will run a model on laptop, then run the unmodified model on the cloud.  On the cloud, we will run hyperparameter sweeps in parallel 8 ways.  The parallel run to **complete the run 8x faster** and  spot instance to **reduce cost of the run by 70%**.  

- Single Run [![Single Run](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/robert-s-lee/grid-optuna/blob/dbb7c20cad6bfb419a037f8ff93cb9774fedb2e5/pytorch_lightning_simple.py&cloud=grid&use_spot&instance=t2.medium&accelerators=1&disk_size=200&framework=lightning&script_args=pytorch_lightning_simple.py)
- 8x Parallel Hyperparameter Sweeps [![Single Run](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/robert-s-lee/grid-optuna/blob/dbb7c20cad6bfb419a037f8ff93cb9774fedb2e5/pytorch_lightning_simple.py&cloud=grid&use_spot&instance=t2.medium&accelerators=1&disk_size=200&framework=lightning&script_args=pytorch_lightning_simple.py%20--pruning="[0,1]"%20--batchsize="[32,128]"%20--epochs="[5,10]")

# Overview

We will use familiar [MNIST](http://yann.lecun.com/exdb/mnist/).
Grid.ai is the creators of PyTorch Lightning.  Grid.ai is agnostics to Machine Learning frameworks and 3rd party tools.
The benefits of Grid.ai are available to other Machine Learning frameworks and tools.
To demonstrate this point, we will NOT use [PyTorch Lightning's Early Stop](https://medium.com/pytorch/pytorch-lightning-1-3-lightning-cli-pytorch-profiler-improved-early-stopping-6e0ffd8deb29).
Instead, we will use [Optuna](https://optuna.org) for early stopping.
We will track progress by viewing [PyTorch Lightning](https://www.pytorchlightning.ai)'s [Tensorboard](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html) in Grid.ai's [Tensorboard interface](https://docs.grid.ai/products/run-run-and-sweep-github-files/metrics-charts#tensorboard).

Grid.ai will launch experiments in parallel using [Grid Search](https://docs.grid.ai/products/run-run-and-sweep-github-files/sweep-syntax) strategy.  Grid.ai Hyperparamter sweep control `batchsize`, `epochs`, `pruning` -- whether Optuna is active or not. Optuna will control the the number of layers, hidden units in each layer and dropouts within each experiment.  The following combinations will result in 8 parallel experiments:

- batchsize=[32,128]
- epochs=[5,10]
- pruning=[0,1]

A single Grid.ai CLI command initiates the experiment.
 
``` bash
grid run --use_spot pytorch_lightning_simple.py --datadir grid:fashionmnist:7 --pruning="[0,1]"  --batchsize="[32,128]" --epochs="[5,10]"
```

# Step by Step Instruction

This instruction assumes access to a laptop with `bash` and `conda`.  For those with restricted local environment, please use SSH on [Grid.ai Session](https://docs.grid.ai/products/sessions#start-a-session).

## Local python environment setup

```bash
# create conda env
conda create --name gridai python=3.7
conda activate gridai
# install packages
pip install lightning-grid
pip install optuna
pip install pytorch_lightning
pip install torchvision
# login to grid
grid login --username <username> --key <grid api key>
```

## Run locally

```bash
# retrieve the model
git clone https://github.com/robert-s-lee/grid-optuna
cd grid-optuna
mkdir data
# Run without Optuna pruning (takes a while)
python pytorch_lightning_simple.py --datadir ./data
# Run with Optuna pruning (takes a while)
python pytorch_lightning_simple.py --datadir ./data --pruning 1
```

## Prepare Grid.ai Datastore 

Setup [Grid.a Datastore](https://docs.grid.ai/products/global-cli-configs/cli-api/grid-datastores) so that MNIST data is not downloaded on each run.  Note the **Version** number created.  Typically this will be **1**.

```bash
grid datastore create --source data --name fashionmnist 
grid datastore list
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Credential Id ┃              Name ┃ Version ┃     Size ┃          Created ┃    Status ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ cc-qdfdk      │      fashionmnist │       1 │ 141.6 MB │ 2021-06-16 15:13 │ Succeeded │
└───────────────┴───────────────────┴─────────┴──────────┴──────────────────┴───────────┘
```
        
## Run on Grid

- Option 1: with Datastore option so that FashionMNIST is not downloaded again (use on your own or with sharable datastore)  
```bash
grid run --use_spot pytorch_lightning_simple.py --datadir grid:fashionmnist:7 --pruning="[0,1]"  --batchsize="[32,128]" --epochs="[5,10]"
```

- Option 2: without Datastore and can be shared freely without creating datastore 
```bash
grid run --use_spot pytorch_lightning_simple.py --pruning="[0,1]"  --batchsize="[32,128]" --epochs="[5,10]"
```

The above commands will show below (abbreviated)
  
```bash
Run submitted!
`grid status` to list all runs
`grid status smart-dragon-43` to see all experiments for this run
```

`grid status smart-dragon-43` shows experiments running in parallel
  
```bash
% grid status smart-dragon-43
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Experiment           ┃                     Command ┃  Status ┃    Duration ┃                  datadir ┃ pruning ┃ batchsize ┃ epochs ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ smart-dragon-43-exp7 │ pytorch_lightning_simple.py │ running │ 0d-00:07:24 │ /datastores/fashionmnist │       1 │        32 │     10 │
│ smart-dragon-43-exp6 │ pytorch_lightning_simple.py │ running │ 0d-00:07:27 │ /datastores/fashionmnist │       1 │        32 │      5 │
│ smart-dragon-43-exp5 │ pytorch_lightning_simple.py │ running │ 0d-00:07:14 │ /datastores/fashionmnist │       1 │       128 │      5 │
│ smart-dragon-43-exp4 │ pytorch_lightning_simple.py │ pending │ 0d-00:12:52 │ /datastores/fashionmnist │       0 │       128 │      5 │
│ smart-dragon-43-exp3 │ pytorch_lightning_simple.py │ running │ 0d-00:07:13 │ /datastores/fashionmnist │       0 │        32 │     10 │
│ smart-dragon-43-exp2 │ pytorch_lightning_simple.py │ running │ 0d-00:07:03 │ /datastores/fashionmnist │       0 │       128 │     10 │
│ smart-dragon-43-exp1 │ pytorch_lightning_simple.py │ running │ 0d-00:07:02 │ /datastores/fashionmnist │       1 │       128 │     10 │
│ smart-dragon-43-exp0 │ pytorch_lightning_simple.py │ pending │ 0d-00:12:52 │ /datastores/fashionmnist │       0 │        32 │      5 │
└──────────────────────┴─────────────────────────────┴─────────┴─────────────┴──────────────────────────┴─────────┴───────────┴────────┘
```

`grid logs smart-dragon-43-exp0` shows logs from that experiment

```bash
grid logs smart-dragon-43
```
## Simpler variations to run

```bash
grid run --use_spot pytorch_lightning_simple.py
grid run --use_spot pytorch_lightning_simple.py --datadir grid:fashionmnist:7"
```

## Use Grid.ai WebUI for Tensorboard graphs

Example of on-demand pricing (top at $0.09) and spot pricing (bottom at $0.03)

![](images/on-demand-spot-cost.png)

Example Metric from Grid.ai WebUI

![](images/grid-val-acc.png)

Example Metric from Tensorboard

![](images/tensorboard-parallel-coord.png)

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

