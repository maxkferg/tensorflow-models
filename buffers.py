import numpy as np


class HistoryBuffer:
    """
    Return blocks of consequetively observed states
    """

    def __init__(self, environment, history_length, state_size):
        """
        """
        self.states = np.zeros((history_length, state_size))
        self.rewards = np.zeros((history_length))
        self.environment = environment
        # Initialize the history
        for _ in range(history_length):
            self._step()


    def fetch(self):
        """
        Fetch the next example
        """
        return self._step()


    def _step(self):
        """
        Step the environment and roll the buffer
        Return (state_history, reward_history)
        """
        action = self.environment.action_space.sample()
        state, reward, done, info = self.environment.step(action)
        # Append to the buffer
        self.rewards = np.roll(self.rewards, shift=-1, axis=0)
        self.states = np.roll(self.states, shift=-1, axis=0)
        self.states[-1,:] = state
        self.rewards[-1] = reward
        # Return state and reward history
        return self.states, self.rewards



