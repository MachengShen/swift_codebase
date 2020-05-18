import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
import time
BENCHMARK = True

"""
def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, benchmark=BENCHMARK, discrete_action=True)
            #env.seed(seed + rank * 1000)
            #np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
"""
def run(config):
    cover_ratio = []

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)
    # logger = SummaryWriter(str(log_dir))

#    torch.manual_seed(run_num)
#    np.random.seed(run_num)
    #env = make_parallel_env(, config.n_rollout_threads, run_num)
    env = make_env(config.env_id, benchmark=BENCHMARK, discrete_action=True)
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)

    model.init_from_save_self('./models/swift_scenario/model/run6/model.pt')
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0

    update_count = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(model.nagents)]

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy().squeeze() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            # agent_actions[0][5]=1
            # agent_actions[1][5]=1
            # agent_actions[2][5]=1
            next_obs, rewards, dones, infos = env.step(agent_actions)
            env.render()
            time.sleep(0.1)


            # # # get actions as torch Variables
            # torch_agent_actions = model.step(torch_obs, explore=True)
            # # convert actions to numpy arrays
            # agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # # rearrange actions to be per environment
            # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            # next_obs, rewards, dones, infos = env.step(actions)
            # env.render()



            #if et_i == config.episode_length - 1:
                #print(infos)
                #print(type(infos['cover_ratio']))
                #cover_ratio.append(float(infos[0]['n'][0]['cover_ratio']))
                #print(infos)



#            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            '''
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):

                    update_count += 1
                    print("episode:", ep_i, ", total steps:", t, " update_count:", update_count)

                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

        logger.export_scalars_to_json(str(log_dir / 'summary.json'))

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    print(cover_ratio)
    '''
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='swift_scenario', help="Name of environment")
    parser.add_argument("--model_name", default='none',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50, type=int)
    parser.add_argument("--episode_length", default=128, type=int)
    parser.add_argument("--steps_per_update", default=1024, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1280000, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=1., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)