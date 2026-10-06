"""Small calendar-year source blocks, with unchanged accepted month rows.

The inherited reader either retains the entire source table or decodes it for
each requested month. This reader retains bounded year blocks instead. Cached
rows outside a requested month are never returned to fitting or selection.
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class CalendarSourcePanels:
    def __init__(self, files, columns, maximum_bytes):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError('Explicit positive source-cache byte limit required')
        self.files = files
        self.columns = {name: tuple(values) for name, values in columns.items()}
        self.maximum_bytes = maximum_bytes
        self.blocks = OrderedDict()
        self.retained_bytes = self.peak_retained_bytes = 0
        self.reads = self.hits = self.fallbacks = self.evictions = 0

    def read(self, name, columns, day):
        path = self.files.path(name)
        if name not in self.columns or not set(columns) <= set(self.columns[name]):
            raise ValueError('Unregistered source panel or columns')
        day = pd.Timestamp(day)
        if pd.isna(day) or day.tz is not None:
            raise ValueError('Timezone-free source date required')
        key = (name, day.year)
        if key in self.blocks:
            self.hits += 1
            self.blocks.move_to_end(key)
            table, dates, size = self.blocks[key]
        else:
            start, stop = pd.Timestamp(day.year, 1, 1), pd.Timestamp(day.year+1, 1, 1)
            table = pq.read_table(path, columns=list(self.columns[name]),
                filters=[('eom','>=',start),('eom','<',stop)], use_threads=False)
            self.files.path(name)
            dates = table.column('eom').to_numpy(zero_copy_only=False).astype('datetime64[ns]',copy=False)
            dates.setflags(write=False)
            # Account for full referenced Arrow buffers, not just slice lengths.
            size = table.get_total_buffer_size() + dates.nbytes
            self.reads += 1
            if size <= self.maximum_bytes:
                while self.blocks and self.retained_bytes + size > self.maximum_bytes:
                    _, (_, _, removed) = self.blocks.popitem(last=False)
                    self.retained_bytes -= removed
                    self.evictions += 1
                self.blocks[key] = (table, dates, size)
                self.retained_bytes += size
                self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
            else:
                # This one uncached block is temporary, not admitted to retention.
                self.fallbacks += 1
        positions = np.flatnonzero(dates == np.datetime64(day.to_datetime64(),'ns'))
        return table.select(list(columns)).take(pa.array(positions)).to_pandas()

    def release(self):
        self.blocks.clear()
        self.retained_bytes = 0

    def diagnostics(self):
        return dict(retained_bytes=self.retained_bytes,peak_retained_bytes=self.peak_retained_bytes,
                    maximum_bytes=self.maximum_bytes,year_reads=self.reads,month_hits=self.hits,
                    oversized_uncached_years=self.fallbacks,evictions=self.evictions,
                    policy='calendar_year_blocks_return_only_requested_month')
