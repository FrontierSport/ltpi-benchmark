<div align="center">


# LTPI: A Benchmark for Long-term Player Identification from Single-Camera Football Video

[[Paper]()]


</div>

>**[LTPI: A Benchmark for Long-term Player Identification from Single-Camera Football Video]()**
>
>Murad Dusov, Vasiliy Chelpanov, Andrey Sakhovskiy, Vadim Linkov, Oleg Durygin, Mikhail Moiseev, Matvey Isupov, Konstantin Mitin, Semen Budennyy
>
>[*TODO: arxiv*]()
>


## About


This is official repository of our work LTPI accepted on [CVSports'26](https://vap.aau.dk/cvsports/) which allows to reproduce results presented in paper. Repository is based on [Tracklab](https://github.com/TrackingLaboratory/tracklab) framework and uses some of [SoccerNet](https://github.com/SoccerNet/sn-gamestate) modules that is also based on Tracklab. Besides we integrate Koshkina's [jersey number recognition pipeline](https://github.com/mkoshkina/jersey-number-pipeline) as Tracklab module and implement team classification and simple identification modules. Due to lack of time final part of paper is represented in form of python scripts in corresponding directory, for detailed launch instructions see section "Run" below.

## Run
### Clone this repository:
```bash
git clone ?
```
### Install it using conda:
```bash
conda create -n ltpi pip python=3.10 -y
conda activate ltpi
cd ltpi-benchmark
pip install -e .
```
### Download dataset:
Download LTPI dataset from ... and set source_path in ltpi.yaml

### Run first part of pipeline:
```bash
python -m tracklab.main -cn ltpi
```

### Run second part of pipeline:
Outputs of first part will be located in outputs/ltpi/{run date}/{run time}/states/TODO.pklz. Copy this path and run the following command.
```bash
```


## Citation
TODO

