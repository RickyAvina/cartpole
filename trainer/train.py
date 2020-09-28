import numpy as np
import copy


total_timesteps, total_eps = 0, 0


def train(agent, env):
	while True:
		# Measure performance for reporting results
		
		# Collect one trajectory
		collect_one_trajectory(agent=agent, env=env)

		# Update policy
		agent.update_policy()
	

def collect_one_trajectory(agent, env):
	global total_timesteps
	global total_eps

	ep_reward = 0
	ep_timesteps = 0
	env_observation = env.reset()

	while True:
		env.render()
		# Select Action
		agent_action, log_prob = agent.select_stochastic_action(np.array(env_observation))
		
		# Take action
		new_env_observation, env_reward, done, _ = env.step(copy.deepcopy(agent_action)) 
		
		# Add experience to memory
		agent.add_memory(
			obs=env_observation,
			new_obs=new_env_observation,
			action=agent_action,
			reward=env_reward,
			log_prob=log_prob,
			done=done)
	
		# Set vars for next timestep
		env_observation = new_env_observation
		ep_timesteps += 1
		ep_reward += env_reward
	
		if done:
			total_eps += 1
			# log here
			
			return ep_reward	
