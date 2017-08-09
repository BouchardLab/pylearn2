for i in {399..400}; do
  sbatch run_shared.sh "config.json" "gp33" "$SAVE" "high gamma, alpha" "1." "False"
done
