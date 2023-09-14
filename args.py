
"""
CLI args for the various routines
"""
# Import the argparse library for command-line argument parsing
import argparse

# Define a function to get command-line arguments needed for training
def get_train_args():
    """Get arguments needed in train.py."""
    
    # Create an ArgumentParser object for parsing command-line arguments
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    # Add common arguments used across scripts (defined in add_common_args)
    add_common_args(parser)
    
    # Add training and testing related arguments (defined in add_train_test_args)
    add_train_test_args(parser)

    # Add specific training-related arguments
    parser.add_argument('--eval_steps',
                        type=int,
                        default=8000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='NLL',
                        choices=('NLL', 'acc', 'AUROC'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--model_type',
                        type=str,
                        default = "baseline",
                        choices=("baseline", "visualbert"),
                        help='Model choice for training')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set 'args.maximize_metric' based on the chosen evaluation metric
    if args.metric_name == 'NLL':
        args.maximize_metric = False  # Minimize negative log-likelihood
    elif args.metric_name in ('EM', 'F1'):
        args.maximize_metric = True   # Maximize EM or F1
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    # Return the parsed arguments
    return args

# Define a function to add common arguments used across scripts
def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    
    # Add file paths and paths to image and text models
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./hateful_memes/train.jsonl')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./hateful_memes/dev.jsonl')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./hateful_memes/test.jsonl')
    parser.add_argument('--img_folder_rel_path',
                        type=str,
                        default='./hateful_memes/')
    parser.add_argument('--text_model_path',
                        type=str)    

# Define a function to add common training and testing arguments
def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    
    # Add arguments related to identifying runs, batch size, model loading, etc.
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=1000,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

# Define a function to get command-line arguments needed for testing
def get_test_args():
    """Get arguments needed in test.py."""
    
    # Create an ArgumentParser object for parsing command-line arguments
    parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    # Add common arguments used across scripts (defined in add_common_args)
    add_common_args(parser)
    
    # Add training and testing related arguments (defined in add_train_test_args)
    add_train_test_args(parser)

    # Add specific testing-related arguments
    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')
    parser.add_argument('--model_type',
                        type=str,
                        default = "baseline",
                        choices=("baseline", "visualbert"),
                        help='Model choice for training')
    parser.add_argument('--ensemble_list',
                        type=str,
                        nargs = "+",
                        help='Model best path tars for ensemble',
                        default = [])

    # Check if either a model load path or a list of ensemble models is provided
    args = parser.parse_args()
    if not args.load_path and not args.ensemble_list:
        raise argparse.ArgumentError('Missing required argument --load_path or --ensemble_list')

    # Return the parsed arguments
    return args
