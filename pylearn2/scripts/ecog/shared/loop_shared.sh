for i in {399..400}; do
  sbatch run_shared.sh "config.json" "GP31" "$SAVE" "high gamma" "amplitude" ".5"
done
