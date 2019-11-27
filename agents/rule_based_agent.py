from agents.agent import Agent


class RuleBasedAgent(Agent):
    def __init__(self, args, env, rule_set, id):
        super(RuleBasedAgent, self).__init__(args, env, id)
        self.rule_set = rule_set

    def _get_prediction(self, states, actions=None):
        return self.rule_set(states)

    def _get_action(self, prediction):
        return prediction['a']

    def _copy_shared_model_to_local(self):
        pass
