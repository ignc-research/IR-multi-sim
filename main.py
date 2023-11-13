from argparse import ArgumentParser
from scripts.training.training import train
from scripts.training.evaluation import eval


if __name__ == '__main__':
    parser = ArgumentParser("IR-Multi-Sim", description="Train complex training instructions for machine learning environments with a simple interface")

    # allow parsing file
    parser.add_argument('-f', '--file', help="Environment config file")
    parser.add_argument('-t', '--timesteps', default=1000000, type=int, help="Amount of timesteps to train the model")
    parser.add_argument('-r', '--reset', action="store_true", help="Start model training from scratch")
    parser.add_argument('--eval', action="store_true", help="Start evaluating the specified model")

    # parse arguments
    args = parser.parse_args()

    # path to config file was specified
    if args.file is None:        
        print("Use -f to specify a file path to the yaml config file")
        exit(0)
    
    if args.eval:
        # evaluate model
        eval(args.file, args.timesteps)
    else:
        # train model
        train(args.file, args.timesteps, args.reset)
