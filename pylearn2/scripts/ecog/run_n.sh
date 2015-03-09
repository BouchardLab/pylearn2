#!/bin/bash

for i in `seq 1 $1`;
do
	THEANO_FLAGS='device=gpu' python $2
done
