from gymnasium import Wrapper

class DictActionWrapper(Wrapper):
    def __init__(self, env):
        super(DictActionWrapper, self).__init__(env)

    def step(self, action):
        new_action = [{"step_size": action[i]} for i in range(len(action))]
        return self.env.step(new_action)