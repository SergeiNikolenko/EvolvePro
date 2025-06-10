#!/bin/bash
# Configuration values for SLURM job submission.
# One leading hash ahead of the word SBATCH is not a comment, but two are.
#SBATCH --time=12:00:00 
#SBATCH --job-name=proteinbert_optimal
#SBATCH -n 12
#SBATCH -N 1   
#SBATCH --cpus-per-task=5  
#SBATCH --mem=160gb  
#SBATCH --output out/proteinbert_optimal-%j.out 

source ~/.bashrc
conda activate evolvepro
module load openmind8/gnu-parallel/20240222

datasets=("brenan" "stiffler" "doud" "haddox" "giacomelli" "jones" "kelsic" "lee" "markin" "zikv_E" "cas12f" "cov2_S")

# Function to run dms_main for a given dataset
run_dms_main() {
    dataset_name=$1
    output_file="out/${dataset_name}-proteinbert_optimal.out"

    echo "Running ${dataset_name} dataset:" > ${output_file}
    python3 -u dms_main.py \
        --dataset_name ${dataset_name} \
        --experiment_name "proteinbert_optimal" \
        --model_name "proteinbert" \
        --embeddings_path "/orcd/archive/abugoot/001/Projects/Matteo/Github/EvolvePro/output/plm/proteinbert" \
        --labels_path "/orcd/archive/abugoot/001/Projects/Matteo/Github/EvolvePro/output/dms" \
        --num_simulations 10 \
        --num_iterations 10 \
        --measured_var "activity" \
        --learning_strategies "topn" \
        --num_mutants_per_round 16 \
        --num_final_round_mutants 16 \
        --first_round_strategies "random" \
        --embedding_types "embeddings" \
        --regression_types "randomforest" \
        --embeddings_file_type "csv" \
        --output_dir "/orcd/archive/abugoot/001/Projects/Matteo/Github/EvolvePro/output/dms_results" \
        >> ${output_file} 2>&1
    echo "Done running ${dataset_name} dataset:" >> ${output_file}
}

# Export the function so it's available to GNU Parallel
export -f run_dms_main

# Use GNU Parallel to run the dms_main function in parallel for each dataset
parallel -j12 run_dms_main ::: "${datasets[@]}"