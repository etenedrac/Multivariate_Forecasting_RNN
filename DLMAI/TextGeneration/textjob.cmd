#!/bin/bash
# @ job_name = textjob
# @ initialdir = /gpfs/projects/bsc28/bsc28642/Wind/text
# @ output = job_text%j.out
# @ error = job_text%j.err
# @ total_tasks = 1
# @ gpus_per_node = 1
# @ cpus_per_task = 1
# @ features = k80
# @ wall_clock_limit = 47:50:00

module purge
module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python TextGenerator.py --verbose --progress --save --iterations 15 --neurons 512 --layers 2 --exlen 100 --step 2 --datafile 4