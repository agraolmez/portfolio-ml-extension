"""Bounded immutable Arrow panels avoid repeatedly decoding whole source files."""
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from .memory import charged_bytes


class SourcePanels:
    def __init__(self,files,columns,maximum_bytes):
        if type(maximum_bytes) is not int or maximum_bytes<0:raise ValueError('Explicit source panel memory bound required')
        self.files=files;self.columns={k:tuple(v) for k,v in columns.items()};self.maximum_bytes=maximum_bytes
        self._panels={};self.retained_bytes=0;self.reads=0;self.hits=0;self.fallbacks=0

    def read(self,name,columns,day):
        path=self.files.path(name)
        if name not in self.columns or not set(columns)<=set(self.columns[name]):raise ValueError('Unregistered panel or columns')
        requested=np.datetime64(pd.Timestamp(day).to_datetime64(),'ns')
        if name not in self._panels:
            parquet=pq.ParquetFile(path);schema=parquet.schema_arrow
            widths=[]
            for col in self.columns[name]:
                kind=schema.field(col).type
                if not (pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_boolean(kind) or pa.types.is_timestamp(kind)):
                    raise ValueError('Only fixed-width accepted source columns can be cached')
                widths.append(max(1,kind.bit_width//8))
            # Includes null bitmaps, the month-index vector and allocation headroom.
            estimate=int(1.1*parquet.metadata.num_rows*(sum(widths)+len(widths)/8+8))
            if (self.retained_bytes+estimate<=self.maximum_bytes and
                    charged_bytes()+estimate<20*2**30 and
                    psutil.virtual_memory().available-estimate>4*2**30):
                table=parquet.read(columns=list(self.columns[name]),use_threads=False)
                dates=table.column('eom').to_numpy(zero_copy_only=False).astype('datetime64[ns]',copy=False)
                dates.setflags(write=False);size=table.nbytes+dates.nbytes
                if self.retained_bytes+size>self.maximum_bytes:raise MemoryError('Source panel memory accounting exceeded')
                self.files.path(name)
                self._panels[name]=(table,dates);self.retained_bytes+=size;self.reads+=1
            else:
                self.fallbacks+=1
                return pd.read_parquet(path,columns=list(columns),filters=[('eom','==',pd.Timestamp(day))])
        else:self.hits+=1
        table,dates=self._panels[name]
        positions=np.flatnonzero(dates==requested)
        # Preserve original row order, Arrow-to-pandas types and requested columns.
        return table.select(list(columns)).take(pa.array(positions)).to_pandas()

    def diagnostics(self):
        return dict(retained_bytes=self.retained_bytes,maximum_bytes=self.maximum_bytes,
            whole_source_reads=self.reads,month_hits=self.hits,direct_fallbacks=self.fallbacks)

    def release(self):
        self._panels.clear();self.retained_bytes=0
