"""하나의 실행 deadline. 파일/배치/프레임 경계에서 협력적으로 중단한다."""
import time


class BudgetStop(RuntimeError):
    def __init__(self, stage):
        self.stage = stage
        super().__init__(stage)


class Deadline:
    def __init__(self, seconds=None, *, clock=None, started=None):
        if seconds is not None and seconds <= 0:
            raise ValueError('TIME_LIMIT')
        self.clock = clock or time.perf_counter
        self.started = self.clock() if started is None else started
        self.ends = None if seconds is None else self.started + seconds
        self.stage = 'start'

    def check(self, stage):
        self.stage = stage
        if self.ends is not None and self.clock() >= self.ends:
            raise BudgetStop(stage)

    def elapsed(self):
        return self.clock() - self.started


def check(deadline, stage):
    if deadline is not None:
        deadline.check(stage)
