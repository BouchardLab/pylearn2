for s in ec2 ec9 gp31 gp33; do
  run_name=$s"_hg"
  mkdir -p $SAVE/exps/$run_name
  mkdir -p $SAVE/output/$run_name
  for i in {1..400}; do
    exp_name=$run_name"_"$i
    export THEANO_FLAGS="floatX=float32,base_compiledir=$HOME/.theano/$exp_name,openmp=True,blas.ldflags=-lmkl_rt,allow_gc=False"
  #                                                                  randomize pca avg_ff avg_1f ds
     python -u run_random.py $i "config.json" $s $SAVE "high gamma, gamma" "1." > $SAVE/output/$run_name/$i".o"
  done
done
