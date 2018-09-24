for i in {1..5}; do
  sbatch run_shared.sh "config.json" "gp33" "$SAVE" "alpha" "1." "False" "True"
done
