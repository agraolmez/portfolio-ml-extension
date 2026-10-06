"""The on-disk log is authoritative; a disconnected chat echo is optional."""
import errno
import json
import sys
from temporal_extension.training import _json_value


class DurableLog:
    def __init__(self, destination, *, echo=None):
        self.destination = destination
        self.echo = sys.stdout if echo is None else echo

    def emit(self, value):
        line = json.dumps(value, default=_json_value, allow_nan=False)+'\n'
        self.destination.write(line)
        self.destination.flush()
        if self.echo is None:
            return
        try:
            self.echo.write(line)
            self.echo.flush()
        except OSError as exc:
            if not isinstance(exc, BrokenPipeError) and exc.errno != errno.EPIPE:
                raise
            self.echo = None
        except ValueError:
            if not getattr(self.echo, 'closed', False):
                raise
            self.echo = None
