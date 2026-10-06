"""User-authorized capacity increase; tolerate isolated OS pressure warnings."""
import psutil


class MemoryPolicy:
    def __init__(self):
        self.critical_since=None
        self.low_since=None

    def check(self,elapsed,pressure,available_gib):
        if pressure not in (1,2,4):
            raise MemoryError('Unknown native memory pressure; cannot assess machine state')
        self.critical_since=(elapsed if self.critical_since is None else self.critical_since) if pressure==4 else None
        self.low_since=(elapsed if self.low_since is None else self.low_since) if available_gib<1. else None
        if self.critical_since is not None and elapsed-self.critical_since>=10.:
            raise MemoryError('Critical macOS memory pressure persisted for ten seconds')
        if self.low_since is not None and elapsed-self.low_since>=10.:
            raise MemoryError('Available system memory remained below one GiB for ten seconds')


def admit(size):
    """Retained buffers leave three GiB free and four GiB below the hard ceiling."""
    return (psutil.virtual_memory().available-size>3*2**30
            and psutil.Process().memory_info().rss+size<24*2**30)
