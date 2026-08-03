import contextlib
import io
import sys


class WorkingLog:
    def __init__(self):
        self.buffer = io.StringIO()
        self.raw_output = ""

    @contextlib.contextmanager
    def capture(self, capture_output=True):
        if not capture_output:
            yield self
            return
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = self.buffer
        sys.stderr = self.buffer
        try:
            yield self
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.raw_output = self.buffer.getvalue()

    def get_log(self):
        return self.raw_output
