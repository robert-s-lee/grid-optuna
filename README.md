

## make location of training data command line argument 

```bash
curl -O https://raw.githubusercontent.com/optuna/optuna-examples/main/pytorch/pytorch_lightning_simple.py
chmod a+x pytorch_lightning_simple.py

diff pytorch_lightning_simple.py ~/github/optuna-examples/pytorch/pytorch_lightning_simple.py > patchfile.patch
129c129
<     datamodule = FashionMNISTDataModule(data_dir=args.datadir, batch_size=BATCHSIZE)
---
>     datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
155d154
<     parser.add_argument('--datadir', default=f'{os.getcwd()}', type=str)
```

## run on local  

```bash
mkdir data
pip install optuna
pip install pytorch_lightning
pip install torchvision
python pytorch_lightning_simple.py --datadir ./data
python pytorch_lightning_simple.py --datadir ./data --pruning
```

## prepare to run on Grid  

- setup datastore for repeat run on VMs  

```bash
grid datastore create --source data --name fashionmnist 
```

- setup `requirements.txt`  

```bash
touch requirements.txt
grid sync-env
git add requirements.txt
git commit -m "requirements.txt synced with current environment"
```
        
## Run on Grid

- run without Optuna `pruning`
  
```bash
grid run pytorch_lightning_simple.py --datadir grid:fashionmnist:7
```

- run WITH Optuna `pruning`

```bash
grid run pytorch_lightning_simple.py --datadir grid:fashionmnist:7 --pruning  
```

## Check progress

- The above commands will show below (abbreviated)
  
```bash
                Run submitted!
                `grid status` to list all runs
                `grid status mini-swan-563` to see all experiments for this run
```
- `grid status mini-swan-563` shows experiments running in parallel
  
```bash
━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Experiment         ┃                     Command ┃  Status ┃    Duration ┃                  datadir ┃ pruning ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ mini-swan-563-exp0 │ pytorch_lightning_simple.py │ pending │ 0d-00:03:36 │ /datastores/fashionmnist │    True │
└────────────────────┴─────────────────────────────┴─────────┴─────────────┴──────────────────────────┴─────────┘
```

- `grid logs mini-swan-563-exp0` shows logs from that experiment

```bash
grid logs mini-swan-563-exp0
```