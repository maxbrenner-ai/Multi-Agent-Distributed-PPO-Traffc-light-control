class SimpleTimerRuleSet:
    def __init__(self, params):
        assert 'length' in params, 'Length not in params'
        self.timer_length = params['length']
        self.reset()

    def reset(self):
        self.current_phase_timer = 0

    def __call__(self, state):
        # Can ignore the actual env state
        if self.current_phase_timer >= self.timer_length:
            self.current_phase_timer = 0
            return 1  # signals to the env to switch the actual phase
        self.current_phase_timer += 1
        return 0
