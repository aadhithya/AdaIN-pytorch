#!/bin/bash

run_id=`cat .runid`
echo "Starting Tensorboard with run_id $run_id"

tensorboard --logdir=./.temp/$run_id
