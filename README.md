
```bash
# make location of training data command line argument 
curl -O https://raw.githubusercontent.com/optuna/optuna-examples/main/pytorch/pytorch_lightning_simple.py
chmod a+x pytorch_lightning_simple.py

diff pytorch_lightning_simple.py ~/github/optuna-examples/pytorch/pytorch_lightning_simple.py
129c129
<     datamodule = FashionMNISTDataModule(data_dir=args.datadir, batch_size=BATCHSIZE)
---
>     datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
155d154
<     parser.add_argument('--datadir', default=f'{os.getcwd()}', type=str)
```

```bash
# run on local  
mkdir data
pip install optuna
pip install pytorch_lightning
pip install torchvision
python pytorch_lightning_simple.py --datadir ./data
python pytorch_lightning_simple.py --datadir ./data --pruning
```

```bash
# setup datastore for repeat run on VMs  
grid datastore create --source data --name fashionmnist 

# setup datastore for repeat run on VMs  

% grid datastore list
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Credential Id ┃              Name ┃ Version ┃     Size ┃          Created ┃    Status ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ cc-qdfdk      │      fashionmnist │       7 │ 141.6 MB │ 2021-06-16 15:13 │ Succeeded │

```

Interesting things to note:
- checks for uncommiteed files (which we can safely README.md)
- scanned code and automagically will setup the environment with `torchvision`, `packaging`, `torch` and `pytorch_lightning` which had to be done manually on laptop.  But it did not detect Optuna.  
- `grid:fashionmnist:7` is a short hand for fashionmnist datastore version 7

```bash
grid run pytorch_lightning_simple.py --datadir grid:fashionmnist:7

WARNING Neither a CPU or GPU number was specified. 1 CPU will be used as a default. To use N GPUs pass in '--grid_gpus N' flag.


    WARNING

    The following files are uncommited. Changes made to these
    files will not be avalable to Grid when running an Experiment.

      optuna/README.md

    Would you like to continue?

    You can use the flag --ignore_warnings to skip this warning.
    See details at: https://docs.grid.ai

    [y/N]: y


        WARNING
        No requirements.txt or environment.yml found but we identified below
        dependencies from your source. Your build could crash or not
        start.

        torchvision
        packaging
        torch
        pytorch_lightning


                Run submitted!
                `grid status` to list all runs
                `grid status micro-rhino-434` to see all experiments for this run

                ----------------------
                Submission summary
                ----------------------
                script:                  pytorch_lightning_simple.py
                instance_type:           t2.medium
                use_spot:                False
                cloud_provider:          aws
                cloud_credentials:       cc-qdfdk
                grid_name:               micro-rhino-434
                datastore_name:          None
                datastore_version:       None
                datastore_mount_dir:     None
```

## examine logs
```bash
grid logs micro-rhino-434-exp0

[build] [2021-06-16T16:06:16.977619+00:00] #1 [internal] load build definition from Dockerfile
[build] [2021-06-16T16:06:16.979242+00:00] #1 sha256:1825d53e85b7a8fac7bc90c8b8ae272c48f662d831f45641dfdf0c8330c7c304
[build] [2021-06-16T16:06:16.980404+00:00] #1 transferring dockerfile: 919B done
[build] [2021-06-16T16:06:16.981895+00:00] #1 DONE 0.0s
[build] [2021-06-16T16:06:16.983453+00:00]
[build] [2021-06-16T16:06:16.984650+00:00] #2 [internal] load .dockerignore
[build] [2021-06-16T16:06:16.985894+00:00] #2 sha256:a947a94cc047b61bad71b28f65f1deee5a07498d1f108cf45ab040462b4c36b0
[build] [2021-06-16T16:06:16.987185+00:00] #2 transferring context: 1.40kB done
[build] [2021-06-16T16:06:16.988527+00:00] #2 DONE 0.0s
[build] [2021-06-16T16:06:16.989712+00:00]
[build] [2021-06-16T16:06:16.991148+00:00] #4 [auth] sharing credentials for ******
[build] [2021-06-16T16:06:16.992542+00:00] #4 sha256:9d2ebd0002fc1e34434aac87106a55adc022f61027ab10f954570c9f29ae8ae5
[build] [2021-06-16T16:06:17.100981+00:00] #4 DONE 0.0s
[build] [2021-06-16T16:06:17.102626+00:00]
[build] [2021-06-16T16:06:17.104220+00:00] #3 [internal] load metadata for ******/*******************/grid-images__cpu-ubuntu18.04-py3.8-torch1.7.1-pl1.2.1:manual-v11
[build] [2021-06-16T16:06:17.105775+00:00] #3 sha256:4540001f5a12c554b7786650ec54b1b8863d8fff62177230e535c46a72d24e8a
[build] [2021-06-16T16:06:17.107453+00:00] #3 DONE 0.2s
[build] [2021-06-16T16:06:17.216082+00:00]
[build] [2021-06-16T16:06:17.217481+00:00] #5 [1/5] FROM ******/*******************/grid-images__cpu-ubuntu18.04-py3.8-torch1.7.1-pl1.2.1:manual-v11@sha256:181b4da827cd281228f4031893d27b848c1d3fb082de98ab3063def79397987b
[build] [2021-06-16T16:06:17.218839+00:00] #5 sha256:84322125a6416abfae49b8b9db1fb113872d038a42a582c13c0230622eac03cc
[build] [2021-06-16T16:06:17.220446+00:00] #5 DONE 0.0s
[build] [2021-06-16T16:06:17.222054+00:00]
[build] [2021-06-16T16:06:17.223542+00:00] #8 [internal] load build context
[build] [2021-06-16T16:06:17.225096+00:00] #8 sha256:e030178afb2dffc8292b05736a502cab91502680287207dc0fef9e2744c86fb2
[build] [2021-06-16T16:06:17.226548+00:00] #8 transferring context: 91.24kB 0.0s done
[build] [2021-06-16T16:06:17.228123+00:00] #8 DONE 0.0s
[build] [2021-06-16T16:06:17.229524+00:00]
[build] [2021-06-16T16:06:17.230982+00:00] #6 [2/5] RUN mkdir -p /gridai/project
[build] [2021-06-16T16:06:17.232516+00:00] #6 sha256:81fd188b4a24485284ab4b0b9e5fdd7224a9d3c54acc4194937d11c1253a1877
[build] [2021-06-16T16:06:17.234031+00:00] #6 CACHED
[build] [2021-06-16T16:06:17.235475+00:00]
[build] [2021-06-16T16:06:17.236867+00:00] #7 [3/5] WORKDIR /gridai/project
[build] [2021-06-16T16:06:17.238500+00:00] #7 sha256:9760fc3a0d9095e7bce9c21b3e431b6ca9a5bd5f6297f141f6ecef67052ffe70
[build] [2021-06-16T16:06:17.239989+00:00] #7 CACHED
[build] [2021-06-16T16:06:17.241341+00:00]
[build] [2021-06-16T16:06:17.242814+00:00] #9 [4/5] COPY / /gridai/project/
[build] [2021-06-16T16:06:17.244444+00:00] #9 sha256:850229ddf2c77813c2a6791d2185ad4d6610b17425c07abcc157405d3414952b
[build] [2021-06-16T16:06:17.246076+00:00] #9 DONE 0.1s
[build] [2021-06-16T16:06:17.366586+00:00]
[build] [2021-06-16T16:06:17.367936+00:00] #10 [5/5] RUN echo "Beginning Project Specific Installations" &&     echo "Finished Project Specific Installations"
[build] [2021-06-16T16:06:17.369440+00:00] #10 sha256:e8eaedba3a13139fba3923916df10a221d86731216499a432648568024ff510e
[build] [2021-06-16T16:06:17.789082+00:00] #10 0.453 Beginning Project Specific Installations
[build] [2021-06-16T16:06:17.790587+00:00] #10 0.453 Finished Project Specific Installations
[build] [2021-06-16T16:06:17.792054+00:00] #10 DONE 0.5s
[build] [2021-06-16T16:06:17.793486+00:00]
[build] [2021-06-16T16:06:17.794894+00:00] #11 exporting to image
[build] [2021-06-16T16:06:17.796368+00:00] #11 sha256:e8c613e07b0b7ff33893b694f7759a10d42e180f2b4dc349fb57dc6b71dcab00
[build] [2021-06-16T16:06:17.797836+00:00] #11 exporting layers 0.1s done
[build] [2021-06-16T16:06:17.799310+00:00] #11 writing image sha256:1bc668d0a4dcc2c9de9481edae93ae20148b64f982fb51fbed9a0e69166177af
[build] [2021-06-16T16:06:17.800880+00:00] #11 writing image sha256:1bc668d0a4dcc2c9de9481edae93ae20148b64f982fb51fbed9a0e69166177af done
[build] [2021-06-16T16:06:17.802281+00:00] #11 naming to ******/*******************/robert-s-lee__argecho-cpu:8bc30be33edcb5b51d5cb6823954c432b163600d-f6aedbcc5bbd6b76cfdd792164f4b9bf done
[build] [2021-06-16T16:06:17.803881+00:00] #11 DONE 0.1s
[build] [2021-06-16T16:06:20.152128+00:00] 8bc30be33edcb5b51d5cb6823954c432b163600d-f6aedbcc5bbd6b76cfdd792164f4b9bf: digest: sha256:db544a35f168cfbe0209df6838b694659d5edb9713aa


```