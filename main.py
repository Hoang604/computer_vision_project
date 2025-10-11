# main.py
import argparse
import yaml
from types import SimpleNamespace
import sys
import os
import logging

# --- Configure Logging ---
# Replace print statements with a more professional logging system.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout
)

# --- Handle Imports ---
# Retain this logic to ensure the script can run directly without a setup.py file.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from scripts.runners.rrdb_runner import run_rrdb_training
    from scripts.runners.diffusion_rrdb_refine_runner import run_diffusion_rrdb_refine_training
    from scripts.runners.diffusion_no_refine_runner import run_diffusion_no_refine_training
    from scripts.runners.diffusion_bicubic_refine_runner import run_diffusion_bicubic_refine_training
    from scripts.runners.rectified_flow_runner import run_rectified_flow_training
except ImportError as e:
    logging.error(
        f"Import Error: {e}. Please ensure you are running main.py from the project's root directory.")
    sys.exit(1)

# --- Model and Runner Mapping ---
# Group runners into a data structure for easy management.
MODEL_RUNNERS = {
    'rrdb': run_rrdb_training,
    'diffusion_rrdb_refine': run_diffusion_rrdb_refine_training,
    'diffusion_no_refine': run_diffusion_no_refine_training,
    'diffusion_bicubic_refine': run_diffusion_bicubic_refine_training,
    'rectified_flow': run_rectified_flow_training
}

# --- Command Handling Functions ---


def handle_train(args: argparse.Namespace):
    """
    Handles the logic for the 'train' command.
    """
    logging.info(f"Received 'train' command for model: '{args.model}'")

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: '{args.config}'")
        return
    except Exception as e:
        logging.error(f"Failed to read or parse the configuration file: {e}")
        return

    runner = MODEL_RUNNERS.get(args.model)
    if runner:
        logging.info(f"--- Starting training for model '{args.model}' ---")
        runner(config)
        logging.info(f"--- Finished training for model '{args.model}' ---")
    else:
        # This case rarely occurs due to 'choices' in argparse, but it's good practice to have it.
        logging.error(f"Invalid model '{args.model}' specified.")


def handle_help(args: argparse.Namespace):
    """
    Handles the logic for the 'help' command.
    """
    command_name = args.topic
    if command_name == 'train':
        print("\n------------------------------------------------------------------------------------")
        print("                           DETAILED GUIDE FOR THE 'train' COMMAND")
        print("------------------------------------------------------------------------------------")
        print("This command handles the training process for all available models.")
        print("You must specify the model to train and provide the path to its configuration file.\n")
        print("Usage Examples:")
        print("\n1. To train the RRDBNet model:")
        print("   python main.py train --model rrdb --config configs/config_rrdb.yaml")
        print("\n2. To train the Diffusion model (refining RRDBNet's output):")
        print("   python main.py train --model diffusion_rrdb_refine --config configs/config_diffusion_rrdb_refine.yaml")
        print("\n3. To train the Rectified Flow model:")
        print("   python main.py train --model rectified_flow --config configs/config_rectified_flow.yaml")
        print("\n------------------------------------------------------------------------------------")
    else:
        print(
            f"\nNo detailed help available for the command '{command_name}'.")

# --- Main Function ---


def main():
    """
    Main function to parse arguments and launch the corresponding commands.
    """
    parser = argparse.ArgumentParser(
        description="A centralized manager for the Computer Vision Super-Resolution project.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest="command", title="Available Commands", metavar="<command>")
    subparsers.required = True

    # --- Define 'train' command ---
    train_parser = subparsers.add_parser(
        "train", help="Starts a training session.")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_RUNNERS.keys(),  # Automatically get the list of models.
        help="The name of the model architecture to train."
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the .yaml configuration file for the selected model."
    )
    # Assign the handler function for this command.
    train_parser.set_defaults(func=handle_train)

    # --- Define 'help' command ---
    help_parser = subparsers.add_parser(
        "help", help="Displays a detailed guide for a command.")
    help_parser.add_argument(
        "topic",
        type=str,
        nargs='?',
        default="train",
        help="The command to get help for (e.g., 'train')."
    )
    help_parser.set_defaults(func=handle_help)

    # --- Handle case when run with no arguments ---
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return

    args = parser.parse_args()
    args.func(args)  # Call the assigned handler function.


if __name__ == "__main__":
    main()
