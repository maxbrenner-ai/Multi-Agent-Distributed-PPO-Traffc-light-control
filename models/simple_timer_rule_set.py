class SimpleTimerRuleSet:
    def __init__(self, params):
        assert 'length' in params, 'Length not in params'
        self.timer_length = params['length']
