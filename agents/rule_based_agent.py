from agents.agent import Agent


class RuleBasedAgent(Agent):
    def __init__(self, args, env, rule_set, data_collector, id):
        super(RuleBasedAgent, self).__init__(args, env, id, data_collector)
        self.rule_set = rule_set

    def _reset(self):
        self.rule_set.reset()

    def _get_prediction(self, states, actions=None):
        assert actions == None  # Doesnt want actions
        assert states.shape[0] == 1, states.shape  # Actually can only handle one state (only needs to)
        return {'a': self.rule_set(states)}

    def _get_action(self, prediction):
        return prediction['a']

    def _copy_shared_model_to_local(self):
        pass
