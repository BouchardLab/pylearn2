for i in {1..400}; do
  sbatch run_shared.sh "config.json" "EC2" "$SAVE" "high gamma" "amplitude"
done
