from argparse import ArgumentParser
from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from scripts.utils.model import setup_model
from scripts.utils.callbacks import parse_callback
import signal

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
        # load model
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
        exit(0)
    
    # train model for desired amount of timesteps
    else:
        # load or create model
        model, model_path = setup_model(model_params, env)
        
        # handle interrupt like ctrl c
        currentTimesteps = 0
        def signal_handler(sig, frame):
            model.save(model_path  + f"/interrupt_at_{model.num_timesteps}.zip")
            print("Training was interrupted!")
            env.close()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

        # train model
        while currentTimesteps < train_parameters["timesteps"]:
            # train model for save_freq steps
            model.learn(train_parameters["save_freq"], callback=parse_callback(train_parameters["logging"], environment_params.get_distance_names()), reset_num_timesteps=False)
            
            # update timesteps
            currentTimesteps += train_parameters["save_freq"]  

            # save model every save_freq steps
            if currentTimesteps % train_parameters["save_freq"] == 0:
                model.save(model_path + "/" + f"checkpoint_{model.num_timesteps}.zip")

        # save final model and clean up environment
        model.save(model_path + "/" + f"checkpoint_{model.num_timesteps}.zip")
        env.close()

        print("Finished training!")
        exit(0)
