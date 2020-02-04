#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "USAGE: create_frames.sh DATASETS_ROOT DATABASE_NAME RESULTS_DIR"
        exit
fi

DATA_ROOT=$1
DB=$2
RESULTSDIR=$3

echo $DATA_ROOT
echo $DB
echo $RESULTSDIR

for epoch in {10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500}
do
echo 'Epoch '$epoch
CMD='--dataroot '$DATA_ROOT'/'$DB' --name '$DB' --model pix2pix --dataset_mode aligned --gpu_id -1 --load_iter '$epoch' --force_test_output '$RESULTSDIR'/'$DB'_'$epoch
echo $CMD
python test_nohtml.py $CMD
done