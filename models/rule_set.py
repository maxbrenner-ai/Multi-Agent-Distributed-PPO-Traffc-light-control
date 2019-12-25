class RuleSet:
    def __init__(self, params, net_path=None, constants=None):
        self.net_path = net_path

    def reset(self):
        raise NotImplementedError

    def __call__(self, state):
        raise NotImplementedError
