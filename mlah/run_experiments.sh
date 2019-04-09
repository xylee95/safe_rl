#!/bin/bash
set -e

savename="mirror_bias_3"
environment="GridWorld-v0"
num_subs=2
augment="True"
num_rollouts=1024
train_time=700
continue_iter="False"
warmup=100
load=""
save="test"
tag="feature_test"


pretrain=0
number=1
for number in {1..5}
do
	filename="${savename}_${environment}_${tag}_augment_${augment}_batchsize_${num_rollouts}_${number}"
	python3 main.py --warmup_time $warmup --filename $filename --pretrain $pretrain --task $environment --num_subs $num_subs --augment $augment --num_rollouts $num_rollouts --train_time $train_time --id_number $number
done

exit 0


#python main.py --filename ip_adv_base_1 --task InvertedPendulum-v2 --num_subs 2 --augment True --pretrain 0 --num_rollouts 2048 --warmup_time 0 --train_time 50 --replay False meta_test
 