for i in {1..1}; do
  sbatch run_shared.sh "config.json" "gp33" "$SAVE" "high gamma" ".9" "True"
done
