from models.rule_set import RuleSet


class SimpleTimerRuleSet(RuleSet):
    def __init__(self, params, net_path=None, constants=None):
        super(SimpleTimerRuleSet, self).__init__(params)
        assert 'length' in params, 'Length not in params'
        self.timer_length = params['length']
        self.offset = params['offset'] if 'offset' in params else None
        self.start_phase = params['start_phase'] if 'start_phase' in params else None
        assert isinstance(self.timer_length, list)
        self.reset()

    # def reset(self):
    #     self.current_phase_timer = 0
    #     self.current_phase = 0

    def reset(self):
        # Start at offset if given, else zeros
        self.current_phase_timer = [x for x in self.offset] if self.offset else [0] * len(self.timer_length)
        self.just_reset = True

    # def __call__(self, state):
    #     # Can ignore the actual env state
    #     if self.current_phase_timer >= self.timer_length:
    #         self.current_phase_timer = 0
    #         if self.current_phase == 0: self.current_phase = 1
    #         elif self.current_phase == 1: self.current_phase = 0
    #         return self.current_phase
    #     self.current_phase_timer += 1
    #     return self.current_phase

    def _convert_to_dec(self, bin):
        return int("".join(str(x) for x in bin), 2)

    def __call__(self, state):
        # If just reset then send switch for start phases of 1
        if self.start_phase and self.just_reset:
            self.just_reset = False
            return self._convert_to_dec(self.start_phase)
        li = []
        for i in range(len(self.timer_length)):
            # Can ignore the actual env state
            if self.current_phase_timer[i] >= self.timer_length[i]:
                self.current_phase_timer[i] = 0
                li.append(1)  # signals to the env to switch the actual phase
                continue
            self.current_phase_timer[i] += 1
            li.append(0)
        # Convert binary list to dec value
        dec = self._convert_to_dec(li)
        return dec
