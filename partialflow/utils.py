import time


class Timer(object):
    def __init__(self):
        self._start = 0
        self._end = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.time()

    @property
    def duration(self):
        if self._end == 0:
            return time.time() - self._start
        else:
            return self._end - self._start


class VerboseTimer(Timer):
    def __init__(self, name):
        super(VerboseTimer, self).__init__()

        self._name = name

    def __enter__(self):
        print('START: %s...' % self._name)
        return super(VerboseTimer, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(VerboseTimer, self).__exit__(exc_type, exc_val, exc_tb)
        print('DONE: %s took %.3f seconds.' % (self._name, self.duration))