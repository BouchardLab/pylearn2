export SAVE=/scratch2/scratchdirs/jlivezey/exps_ff_1f
#for s in ec2 ec9 gp31 gp33; do
for s in gp31; do
  #                                                                  randomize pca avg_ff avg_1f ds
  sbatch -a 1-5 -t 05:40:00 -J $s"_1f_hg_g_ds" run_shared.sh "config.json" $s $SAVE "high gamma, gamma" "1." "False" "False" "False" "True" "True"
done
