#!/usr/bin/env python3
import os
import sys

# This script must be run from the 'models' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) != 'models':
    print('Error: create_grid_search.py must be run from the models/ directory', file=sys.stderr)
    sys.exit(1)

# Determine project root (parent of 'models')
project_root = os.path.dirname(script_dir)

# Hyperparameter grid: layers 10, 12, and 14
LAYER_VALS = [10, 12, 14]
HEAD_VALS  = [8, 10, 12]

# Paths
PARAMS_PATH  = os.path.join(script_dir, 'transformer_setup', 'params.py')
BACKUP_PATH  = os.path.join(script_dir, 'transformer_setup', 'params.py.bak')
SCRIPT_NAME  = 'grid_search.sh'
RESULTS_CSV  = os.path.join(project_root, 'new_results.csv')

layers_str = ' '.join(map(str, LAYER_VALS))
heads_str  = ' '.join(map(str, HEAD_VALS))

# Build bash driver
bash = f"""#!/usr/bin/env bash
set -euo pipefail

# No combinations to skip
SKIP_COMBOS=()

echo "Changing to project root: {project_root}"
cd "{project_root}"

# Only write header if CSV doesn't exist
if [ ! -f "{RESULTS_CSV}" ]; then
  echo "num_layers,num_heads,best_val_loss,avg_loss,perplexity,generated_text" > "{RESULTS_CSV}"
fi

# Backup original params.py
cp "{PARAMS_PATH}" "{BACKUP_PATH}"

for L in {layers_str}; do
  for H in {heads_str}; do

    RUN_DIR="{project_root}/models/runs/L${{L}}_H${{H}}"
    mkdir -p "${{RUN_DIR}}/logs"

    # Patch hyperparameters
    sed -i "s/self.n_layer = .*/self.n_layer = ${{L}}/" "{PARAMS_PATH}"
    sed -i "s/self.n_head  = .*/self.n_head = ${{H}}/" "{PARAMS_PATH}"
    EMB=$(( (H==8||H==12)?768:760 ))
    sed -i "s/self.n_embd = .*/self.n_embd = ${{EMB}}/" "{PARAMS_PATH}"
    sed -i "s#self.log_dir = .*#self.log_dir = '${{RUN_DIR}}/logs'#" "{PARAMS_PATH}"

    echo "=== RUN L=${{L}} H=${{H}} EMB=${{EMB}} ==="

    # Train (cwd=models)
    pushd "{script_dir}" > /dev/null
      python3 -u gpt_original.py 2>&1 | tee "${{RUN_DIR}}/logs/run.log"
    popd > /dev/null

    # Grab the last best-model val loss
    BEST_VAL_LOSS=$(grep "New best model saved with val loss" "${{RUN_DIR}}/logs/run.log" | tail -1 | awk '{{print $NF}}')
    GENERATED_TEXT=$(grep -m1 "Generated text:" "${{RUN_DIR}}/logs/run.log" | sed -E 's/Generated text: //')

    # Evaluate using default checkpoint dir
    CHECKPOINT_PATH="{project_root}/models/checkpoints/best_model.pt"
    pushd "{project_root}/evaluation" > /dev/null
      python3 eval_perplexity.py "$CHECKPOINT_PATH" > "${{RUN_DIR}}/logs/eval.log"
    popd > /dev/null

    # Parse evaluation outputs
    AVG_LOSS=$(grep "Average Loss:" "${{RUN_DIR}}/logs/eval.log" | awk '{{print $3}}')
    PERPLEXITY=$(grep "Perplexity:"    "${{RUN_DIR}}/logs/eval.log" | awk '{{print $2}}')

    # Print metrics to terminal
    echo "L=${{L}} H=${{H}} -> avg_loss=${{AVG_LOSS}}, perplexity=${{PERPLEXITY}}"

    # Copy best_model into the run directory
    cp "$CHECKPOINT_PATH" "${{RUN_DIR}}/best_model.pt"

    # Delete all other checkpoints, keep only best_model.pt
    CHECKPOINT_DIR="{project_root}/models/checkpoints"
    for ckpt in "${{CHECKPOINT_DIR}}"/*; do
      if [[ "$(basename "$ckpt")" != "best_model.pt" ]]; then
        rm -f "$ckpt"
      fi
    done

    # Append results to new_results.csv
    echo "${{L}},${{H}},${{BEST_VAL_LOSS}},${{AVG_LOSS}},${{PERPLEXITY}},\"${{GENERATED_TEXT}}\"" \
      >> "{RESULTS_CSV}"

    # Restore params.py for next iteration
    mv "{BACKUP_PATH}" "{PARAMS_PATH}"
    cp "{PARAMS_PATH}" "{BACKUP_PATH}"
  done
done

# Final restore of params.py
mv "{BACKUP_PATH}" "{PARAMS_PATH}"

echo "✅ Grid search for layers 10, 12, and 14 complete.
Logs and best_model.pt retained in each run directory; other checkpoints cleaned."
"""

# Write the bash script
with open(os.path.join(script_dir, SCRIPT_NAME), 'w') as f:
    f.write(bash)

print(f"✅ Generated {SCRIPT_NAME} in {script_dir}")
print("Run it now from your models directory:")
print("  cd models")
print("  chmod +x grid_search.sh")
print("  nohup ./grid_search.sh > grid_search.log 2>&1 &")