#!/bin/bash

# rsync --progress -a -e ssh mf724021@copy23-1.hpc.itc.rwth-aachen.de:/work/mf724021/federated_learning/ /data/federated_learning/ 

rsync --progress -a -e ssh mf724021@copy23-1.hpc.itc.rwth-aachen.de:/work/p0021919/distributed_transformer/ /data/distributed_transformer/ 

# rsync --progress -a -e ssh /data/datasets/ mf724021@copy23-1.hpc.itc.rwth-aachen.de:/hpcwork/p0021919/datasets/ 
