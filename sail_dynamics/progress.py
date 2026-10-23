"""Console progress display without third-party dependencies."""
import sys


class TrialProgressBar:
    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.active = False

    def update(self, completed, total):
        width = 30
        filled = width*completed//total
        bar = '#'*filled+'-'*(width-filled)
        digits = len(str(total))
        self.active = completed < total
        print(f'\rTrials [{bar}] {completed:>{digits}}/{total} ({100*completed/total:5.1f}%)',
              end='' if self.active else '\n', file=self.stream, flush=True)

    def close(self):
        if self.active:
            print(file=self.stream, flush=True)
            self.active = False
