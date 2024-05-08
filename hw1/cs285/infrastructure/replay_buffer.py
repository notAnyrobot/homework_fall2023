from cs285.infrastructure.utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]


    def batch_generator(self, train_batch_size: int, seq_len: int = None):

        batch_size = self.obs.shape[0]
        assert batch_size >= train_batch_size, (
            "Number of timesteps in replay buffer less than the number requested"
            " for training."
        ) 
        # generate random indices
        indices = np.random.permutation(self.obs.shape[0])
        num_train_batches = int(batch_size // train_batch_size)
        for batch_num in range(num_train_batches):
            start_idx = batch_num * train_batch_size
            end_idx = (batch_num + 1) * train_batch_size
            batch_indices = indices[start_idx:end_idx]

            yield (self.obs[batch_indices], 
                   self.acs[batch_indices],)

            # yield (self.obs[batch_indices],
            #        self.acs[batch_indices],
            #        self.rews[batch_indices],
            #        self.next_obs[batch_indices],
            #        self.terminals[batch_indices])