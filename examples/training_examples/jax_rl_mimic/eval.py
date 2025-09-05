import os
import argparse
import numpy as np

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

from omegaconf import OmegaConf


class MinimalObservationWrapper:
  
    """
    Un wrapper minimale che non fa alcuna assunzione sull'API dell'ambiente,
    a parte l'esistenza dei metodi step() e reset().
    Intercetta i risultati di questi metodi per troncare il vettore di osservazione.
    """
  
    def __init__(self, env, expected_obs_dim):
        self._env = env
        self.expected_obs_dim = expected_obs_dim

    def _truncate_obs(self, obs):
      
        return np.asarray(obs)[:self.expected_obs_dim]

    def step(self, action):
       # Esegue un passo nell'ambiente e tronca l'osservazione restituita.
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._truncate_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        
        """ Resetta l'ambiente e tronca l'osservazione restituita.
        Questa versione gestisce correttamente il fatto che env.reset() e la funzione
        chiamante si aspettano entrambi un solo valore di ritorno (l'osservazione).
        """
        # L'ambiente originale restituisce solo l'osservazione
        obs = self._env.reset(**kwargs)
        
        # Restituiamo solo l'osservazione troncata
        return self._truncate_obs(obs)

    def __getattr__(self, name):
       
        """ Inoltra tutte le altre chiamate di metodi e accessi agli attributi
        all'ambiente originale.
        """
        return getattr(self._env, name)


os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# Applica il nostro wrapper minimale per risolvere il problema della dimensione delle osservazioni
# Usa la dimensione originale di 434 dall'errore del traceback precedente.
OBS_DIM_DURING_TRAINING = 434 
env = MinimalObservationWrapper(env, OBS_DIM_DURING_TRAINING)


# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=10000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True,
                       train_state_seed=0)
