for i in {1..1}; do
  sbatch run_shared.sh "config.json" "ec2" "$SAVE" "high gamma" "amplitude" ".5" "True"
done
