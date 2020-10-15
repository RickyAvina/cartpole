import numpy as np
import copy
import os


total_timesteps, total_eps, running_reward = 0, 0, 0
test = False
    

def test(agent, env, log, tb_writer, args):
    ep_reward = 0
    ep_timesteps = 0
    obs = env.reset()

    while True:
        env.render()
        
        # Select Action
        agent_action, log_prob = agent.select_stochastic_action(np.array(obs))

        # Take action
        # TODO deepcopy should not be needed I guess
        new_obs, env_reward, done, _ = env.step(copy.deepcopy(agent_action))

        # Set vars for next timestep
        obs = new_obs
        ep_timesteps += 1
        ep_reward += env_reward
    
        if done:
            tb_writer.add_scalars("reward", {"train_reward": ep_reward}, 0)
            ep_reward = 0
            obs = env.reset()


def train(agent, env, log, tb_writer, args):
    global total_eps

    while True:
        # Collect one trajectory
        collect_one_trajectory(agent=agent, env=env, log=log, tb_writer=tb_writer, args=args)
        tb_writer.add_scalar("debug/memory", len(agent.memory), total_eps)

        # Update policy
        agent.update_policy(total_eps=total_eps)
                
        # reinitialize memory
        agent.clear_memory()        

        # Save model every 5 updates
        if (running_reward >= 195.0):
            agent.save_weight(os.getcwd() + "/pytorch_models/", "model" + str(total_eps))
            print("Saved model {}!".format(total_eps))


def eval_progress(env, agent, n_eval, log, tb_writer, args):
    raise NotImplementedError

        
def collect_one_trajectory(agent, env, log, tb_writer, args):
    global total_timesteps
    global total_eps
    global running_reward

    ep_reward = 0
    ep_timesteps = 0
    obs = env.reset()

    while True:
        # env.render()
        
        # Select Action
        agent_action, log_prob = agent.select_stochastic_action(np.array(obs))
        
        # Take action
        # TODO I would check whether deepcopy is needed. If action is changed then yes, if not no
        new_obs, env_reward, done, _ = env.step(copy.deepcopy(agent_action)) 

        # Add experience to memory
        agent.add_memory(
            obs=obs,
            new_obs=new_obs,
            action=agent_action,
            reward=env_reward,
            log_prob=log_prob,
            done=done)
        
        # Set vars for next timestep
        obs = new_obs
        ep_timesteps += 1
        ep_reward += env_reward
        total_timesteps += 1
    
        if done:
            total_eps += 1
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            
            tb_writer.add_scalar("reward/train", ep_reward, total_eps)
            tb_writer.add_scalar("reward/running", running_reward, total_eps)
            tb_writer.add_scalars("reward", {"ep_reward": ep_reward}, total_eps)
            tb_writer.add_scalars("reward", {"running_reward": running_reward}, total_eps)
            log[args.log_name].info("Episodic reward: {}".format(ep_reward))
            break
