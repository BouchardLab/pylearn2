for i in {1..400}; do
  sbatch run_shared.sh "config.json" "ec2" "$SAVE" "high gamma, theta" "1." "False"
done
