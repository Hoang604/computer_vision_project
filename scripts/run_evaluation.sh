# setup enviroment 
export TEST_DATA_FOLDER="data/test"  
export PREPROCESSED_DATA_FOLDER="preprocessed_data/rrdb_refined_test"  
export IMG_SIZE=160
export DOWNSCALE_FACTOR=4
export MAX_SAMPLES=100  
export DEVICE="cuda:0"

export RRDB_MODEL_PATH="checkpoints_rrdb/rrdb_model_best.pth"
export DIFFUSION_MODEL_PATH="checkpoints_diffusion/diffusion_model_best.pth"
export RRDB_CONTEXT_PATH="checkpoints_rrdb/context_extractor/rrdb_model_best.pth"

# setup model 
export RRDB_NUM_FEAT=64
export RRDB_NUM_BLOCK=17
export RRDB_GC=32
export UNET_BASE_DIM=64

echo "=== Evaluation Script for Super-Resolution Models ==="
echo "Test data folder: $TEST_DATA_FOLDER"
echo "Max samples: $MAX_SAMPLES"
echo "Device: $DEVICE"
echo ""

# Function to run evaluation
run_evaluation() {
    local eval_type=$1
    local output_file=$2
    local extra_args=$3
    
    echo "Running $eval_type evaluation..."
    python -m scripts.evaluate \
        --test_data_folder "$TEST_DATA_FOLDER" \
        --preprocessed_data_folder "$PREPROCESSED_DATA_FOLDER" \
        --img_size $IMG_SIZE \
        --downscale_factor $DOWNSCALE_FACTOR \
        --max_samples $MAX_SAMPLES \
        --device "$DEVICE" \
        --output_file "$output_file" \
        $extra_args
    
    if [ $? -eq 0 ]; then
        echo "$eval_type evaluation completed successfully!"
        echo "Results saved to: $output_file"
    else
        echo "Error occurred during $eval_type evaluation"
    fi
    echo ""
}

echo "Chọn method để evaluate:"
echo "1) Bicubic baseline only"
echo "2) RRDBNet only" 
echo "3) Diffusion model only"
echo "4) All methods (Bicubic + RRDBNet + Diffusion)"
echo "5) Bicubic + RRDBNet"
echo "6) RRDBNet + Diffusion"
read -p "Nhập lựa chọn (1-6): " choice

case $choice in
    1)
        echo "Evaluating Bicubic baseline..."
        run_evaluation "Bicubic" "bicubic_evaluation.json" "--eval_bicubic"
        ;;
    2)
        echo "Evaluating RRDBNet..."
        run_evaluation "RRDBNet" "rrdb_evaluation.json" "--eval_rrdb --rrdb_model_path '$RRDB_MODEL_PATH' --rrdb_num_feat $RRDB_NUM_FEAT --rrdb_num_block $RRDB_NUM_BLOCK --rrdb_gc $RRDB_GC"
        ;;
    3)
        echo "Evaluating Diffusion model..."
        run_evaluation "Diffusion" "diffusion_evaluation.json" "--eval_diffusion --diffusion_model_path '$DIFFUSION_MODEL_PATH' --rrdb_context_model_path '$RRDB_CONTEXT_PATH' --rrdb_num_feat_context $RRDB_NUM_FEAT --rrdb_num_block_context $RRDB_NUM_BLOCK --rrdb_gc_context $RRDB_GC --unet_base_dim $UNET_BASE_DIM --use_attention --timesteps 1000 --num_inference_steps 50"
        ;;
    4)
        echo "Evaluating all methods..."
        run_evaluation "All" "complete_evaluation.json" "--eval_bicubic --eval_rrdb --eval_diffusion --rrdb_model_path '$RRDB_MODEL_PATH' --diffusion_model_path '$DIFFUSION_MODEL_PATH' --rrdb_context_model_path '$RRDB_CONTEXT_PATH' --rrdb_num_feat $RRDB_NUM_FEAT --rrdb_num_block $RRDB_NUM_BLOCK --rrdb_gc $RRDB_GC --rrdb_num_feat_context $RRDB_NUM_FEAT --rrdb_num_block_context $RRDB_NUM_BLOCK --rrdb_gc_context $RRDB_GC --unet_base_dim $UNET_BASE_DIM --use_attention --timesteps 1000 --num_inference_steps 50"
        ;;
    5)
        echo "Evaluating Bicubic + RRDBNet..."
        run_evaluation "Bicubic+RRDBNet" "bicubic_rrdb_evaluation.json" "--eval_bicubic --eval_rrdb --rrdb_model_path '$RRDB_MODEL_PATH' --rrdb_num_feat $RRDB_NUM_FEAT --rrdb_num_block $RRDB_NUM_BLOCK --rrdb_gc $RRDB_GC"
        ;;
    6)
        echo "Evaluating RRDBNet + Diffusion..."
        run_evaluation "RRDBNet+Diffusion" "rrdb_diffusion_evaluation.json" "--eval_rrdb --eval_diffusion --rrdb_model_path '$RRDB_MODEL_PATH' --diffusion_model_path '$DIFFUSION_MODEL_PATH' --rrdb_context_model_path '$RRDB_CONTEXT_PATH' --rrdb_num_feat $RRDB_NUM_FEAT --rrdb_num_block $RRDB_NUM_BLOCK --rrdb_gc $RRDB_GC --rrdb_num_feat_context $RRDB_NUM_FEAT --rrdb_num_block_context $RRDB_NUM_BLOCK --rrdb_gc_context $RRDB_GC --unet_base_dim $UNET_BASE_DIM --use_attention --timesteps 1000 --num_inference_steps 50"
        ;;
    *)
        echo "Lựa chọn không hợp lệ!"
        exit 1
        ;;
esac

echo "=== Evaluation completed ==="
