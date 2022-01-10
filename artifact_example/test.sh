start_time=$(date +%s)
# perform a task

/home/coder/hdd/miniconda3/envs/torch/bin/python /home/coder/hdd/private/artifact_example/main.py

end_time=$(date +%s)

# elapsed time with second resolution
elapsed=$(( end_time - start_time ))

echo $elapsed