from argparse import ArgumentParser
from scripts.utils.config_parser import parse_config
from scripts.envs import create_env
from scripts.utils.model import setup_model
from scripts.utils.callbacks import parse_callback
import signal
import sys

LOG_DIR = "./logs/csv/"

if __name__ == '__main__':
    parser = ArgumentParser("IR-Multi-Sim", description="Train complex training instructions for machine learning environments with a simple interface")

    # allow parsing file
    parser.add_argument('engine', choices=['pybullet', "isaac"], help="Specify the engine 'pybullet' or 'isaac'")
    parser.add_argument('file', help="Environment config file")
    parser.add_argument('-e', '--eval', action="store_true", help="Start evaluating the specified model")
    
    # extract arguments
    args = parser.parse_args()
    engine = args.engine
    file = args.file
    eval = args.eval

    print(eval)
    # parse config file
    environment_params, model_params, train_parameters, eval_parameters = parse_config(file, engine, eval)     

    # create environment
    env = create_env(environment_params)

    # get csv log path
    csv_path = LOG_DIR + model_params["model_name"]

    # handle interupts
    def signal_handler(sig, frame):
        model.save(model_path  + f"/interrupt_at_{model.num_timesteps}.zip")
        print("Training was interrupted!")
        env.close(csv_path + f"timestep_{model.num_timesteps}")
        exit(0)
  
    # evaluate existing model for desired amount of timesteps
    if eval:
        import time
        # load model
        model_params["load_model"] = True
        model, model_path = setup_model(model_params, env)
        
        obs = env.reset()
        signal.signal(signal.SIGINT, signal_handler)
        for i in range(eval_parameters["timesteps"]):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            time.sleep(0.01)
            print("Step ", i, "; Rewards: ", rewards, "; Dones:", dones)
        
        print("Average Rewards: ", info)
        
        # Print out results and clean up environment
        env.close(csv_path + f"timestep_{i}")
        print("Evaluation finished!")
        exit(0)
    
    # train model for desired amount of timesteps
    else:
        # load or create model
        model, model_path = setup_model(model_params, env)
        
        # handle interrupt like ctrl c
        currentTimesteps = 0        
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
        env.close(csv_path + f"timestep_{model.num_timesteps}")
        print("Finished training!")
        exit(0)
