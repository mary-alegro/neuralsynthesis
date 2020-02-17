#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "USAGE: create_frames.sh DATASET_DIR DATABASE_NAME RESULTS_DIR CHECKPOINTS_DIR GPU(0 or -1 for CPU)"
        exit
fi

DATA_ROOT=$1
DB=$2
RESULTSDIR=$3
CPDIR=$4
GPU=$5

echo $DATA_ROOT
echo $DB
echo $RESULTSDIR

for epoch in {10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500}
do
echo 'Epoch '$epoch
CMD='--dataroot '$DATA_ROOT' --name '$DB' --model pix2pixpl --dataset_mode aligned --gpu_id '$GPU' --load_iter '$epoch' --force_test_output '$RESULTSDIR'/'$DB'_'$epoch' --checkpoints_dir '$CPDIR
echo $CMD
python test_nohtml.py $CMD
done
