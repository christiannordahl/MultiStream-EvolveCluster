MultiStreamEvolveCluster contains all the experiments that have been included in our article (see below for citation).

# Installation
This repository requires an installation of **python 3.10.8** and **poetry** to install the dependencies.

Navigate to the root folder and type:
`poetry install`

This should resolve all python dependencies.

The Markov Clustering algorithm should be installed as per instructions from https://github.com/micans/mcl.
Then put a symlink from the multistreamevolvecluster folder to the mcl binary.

# Datasets
In the article we have used synthetic datasets from our own RBFGenerator, which is located in `multistreamevolvecluster/data/rbgfenerator`.
The data we generated is located in the subfolders `2-dim` and `8-dim`.

Additionally, we used the dataset *The Almanac of Minutely Power Dataset (Version 2)*, which you have to download here.
Place the files Climate_HourlyWeather.csv, Electricity_WHE.csv, NaturalGas_WHG.csv, and Water_WHW.csv, in the folder `multistreamevolveluster/data/AMPds2/original`.
To create the exact data files we used in our experiments, go to the folder `multistreamevolveluster/data/AMPds2` and run the python file **dataconverter.py**.
This creates 1h, 2, 3, 4, 6, and 8h versions of the dataset, divided into 2 month segments.

# Experiments
To then run the experiments, simply run `python MultiStreamEvolveCluster.py`


# Citation

If you intend to use this work, please cite the following paper:

Nordahl, C., Boeva, V., & Grahn, H. (2023). MultiStream EvolveCluster. Proceedings of the Canadian Conference on Artificial Intelligence. https://doi.org/10.21428/594757db.b22e0e9a
