# main.py
import yaml
from types import SimpleNamespace
import sys
import os
import logging

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout
)

# --- Handle Imports ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from scripts.runners.rrdb_runner import run_rrdb_training
    from scripts.runners.diffusion_rrdb_refine_runner import run_diffusion_rrdb_refine_training
    from scripts.runners.diffusion_no_refine_runner import run_diffusion_no_refine_training
    from scripts.runners.diffusion_bicubic_refine_runner import run_diffusion_bicubic_refine_training
    from scripts.runners.rectified_flow_runner import run_rectified_flow_training

    # For interactive inference
    from scripts.rrdb_infer import main as run_rrdb_inference
    from scripts.diffusion_infer import main as run_diffusion_infer
    from scripts.rfpp_super_resolution_infer import main as run_rfpp_inference
except ImportError as e:
    logging.error(
        f"Import Error: {e}. Please ensure you are running main.py from the project's root directory.")
    sys.exit(1)

# --- Model and Runner Mappings ---
MODEL_RUNNERS = {
    'rrdb': run_rrdb_training,
    'diffusion_rrdb_refine': run_diffusion_rrdb_refine_training,
    'diffusion_no_refine': run_diffusion_no_refine_training,
    'diffusion_bicubic_refine': run_diffusion_bicubic_refine_training,
    'rectified_flow': run_rectified_flow_training
}

TRAIN_CONFIG_MAPPING = {
    'rrdb': 'configs/config_rrdb.yaml',
    'diffusion_rrdb_refine': 'configs/config_diffusion_rrdb_refine.yaml',
    'diffusion_no_refine': 'configs/config_diffusion_no_refine.yaml',
    'diffusion_bicubic_refine': 'configs/config_diffusion_bicubic_refine.yaml',
    'rectified_flow': 'configs/config_rectified_flow.yaml'
}

INFERENCE_RUNNERS = {
    'rrdb': run_rrdb_inference,
    'diffusion_rrdb_refine': run_diffusion_infer,
    'rectified_flow': run_rfpp_inference,
}

INFERENCE_CONFIG_MAPPING = {
    'rectified_flow': 'configs/config_rfpp_inference.yaml'
}

# --- Command Handling Functions ---


def handle_train(model_name: str):
    """
    Handles the logic for training a model.
    """
    logging.info(f"Received 'train' command for model: '{model_name}'")

    config_path = TRAIN_CONFIG_MAPPING.get(model_name)
    if not config_path:
        logging.error(
            f"No config file mapping found for model: '{model_name}'")
        return

    logging.info(f"Using configuration file: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: '{config_path}'")
        return
    except Exception as e:
        logging.error(f"Failed to read or parse the configuration file: {e}")
        return

    runner = MODEL_RUNNERS.get(model_name)
    if runner:
        logging.info(f"--- Starting training for model '{model_name}' ---")
        runner(config)
        logging.info(f"--- Finished training for model '{model_name}' ---")
    else:
        logging.error(f"Invalid model '{model_name}' specified.")


def handle_inference(model_name: str):
    """
    Handles the logic for running inference with a model.
    """
    runner = INFERENCE_RUNNERS.get(model_name)
    if runner:
        logging.info(f"--- Starting inference for model '{model_name}' ---")
        config_path = INFERENCE_CONFIG_MAPPING.get(model_name)

        if config_path:
            logging.info(f"Using configuration file: {config_path}")
            runner(config_path)
        else:
            # For runners that don't need a config file
            runner()

        logging.info(f"--- Finished inference for model '{model_name}' ---")
    else:
        logging.error(
            f"Inference for model '{model_name}' is not supported or invalid.")

# --- Main Interactive Function ---


def main():
    """
    Main function to provide an interactive CLI.
    """
    print("Welcome to the CV Super-Resolution project!")

    while True:
        print("\n----------------------------------------------------")
        action = input(
            "What would you like to do? (train/inference/exit): ").lower().strip()

        if action == 'train':
            print("\nSelect a model to train:")
            models = list(MODEL_RUNNERS.keys())
            for i, model in enumerate(models):
                print(f"{i+1}: {model}")

            try:
                choice = input(
                    f"Enter the number of the model (1-{len(models)}): ")
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(models):
                    model_name = models[model_idx]
                    handle_train(model_name)
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif action == 'inference':
            print("\nSelect a model for inference:")
            models = list(INFERENCE_RUNNERS.keys())
            for i, model in enumerate(models):
                print(f"{i+1}: {model}")

            try:
                choice = input(
                    f"Enter the number of the model (1-{len(models)}): ")
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(models):
                    model_name = models[model_idx]
                    handle_inference(model_name)
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif action == 'exit':
            print("Exiting.")
            break

        else:
            print("Invalid action. Please choose 'train', 'inference', or 'exit'.")


if __name__ == "__main__":
    main()
