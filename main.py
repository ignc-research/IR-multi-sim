from argparse import ArgumentParser
from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from scripts.utils.model import setup_model
from scripts.utils.callbacks import parse_callback


if __name__ == '__main__':
    parser = ArgumentParser("IR-Multi-Sim", description="Train complex training instructions for machine learning environments with a simple interface")

    # allow parsing file
    parser.add_argument('-f', '--file', help="Environment config file")
    parser.add_argument('--eval', action="store_true", help="Start evaluating the specified model")
    args = parser.parse_args()
        
    # path to config file was not specified
    if args.file is None:        
        print("Use -f to specify a file path to the yaml config file")
        exit(0)

    # parse config file
    environment_params, model_params, train_parameters, eval_parameters = parse_config(args.file)

    # create environment
    env = create_env(environment_params)
  
    # evaluate existing model for desired amount of timesteps
    if args.eval:
        # load or create model
        model_params["load_model"] = True
        model, model_path = setup_model(model_params, env)
        
        obs = env.reset()
        for i in range(eval_parameters["timesteps"]):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print("Step ", i, "; Rewards: ", rewards, "; Dones:", dones)
        
        print("Average Rewards: ", info)
        
        # Print out results and clean up environment
        env.close()
    
    # train model for desired amount of timesteps
    else:
        # load or create model
        model, model_path = setup_model(model_params, env)

        # train model
        model.learn(train_parameters["timesteps"], callback=parse_callback(train_parameters["logging"], environment_params.get_distance_names()))

        # learning done: safe parameters and clean up environment
        # model.save(model_path)
        env.close()
