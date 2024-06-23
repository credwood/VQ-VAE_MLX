import random
from multiprocessing import Pool

import musdb
import mlx.core as mx
import numpy as np



class DBDataLoader:
    def __init__(self, 
                 batch_size: int = 64, 
                 root_folder: str ="/Users/red/Desktop/musdb18hq", 
                 subsets: str = "train",
                 chunk_duration: float = 5.0,
                 stem: str = "drums",
                 num_processes: int = 4
                 ):
        self.batch_size = batch_size
        self.root_folder = root_folder
        self.subsets = subsets
        self.mus = musdb.DB(root=self.root_folder, is_wav=True, subsets=[self.subsets], sample_rate=44100)
        self.chunk_duration = chunk_duration
        self.stem = stem
        self.num_processes = num_processes
        self.sample_rate = self.mus.sample_rate

    def _make_batch(self, batch_len: int):
        batch = []
        while len(batch) < batch_len:
            track = random.choice(self.mus.tracks)
            track.chunk_duration = self.chunk_duration
            track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
            y = track.targets[self.stem].audio
            num_zeros = np.count_nonzero(y==0)
            non_zeros = np.count_nonzero(y)
            if not non_zeros or (num_zeros/non_zeros) > 0.6:
                continue
            batch.append(mx.array(y))
        return mx.array(batch)
    
    def _start_processed(self):
        num_processes = self.num_processes
        mini_batch_size = int(self.batch_size//num_processes)
        if not mini_batch_size:
            mini_batch_size = 1
            num_processes = self.batch_size
        batches = [(mini_batch_size) for i in range(num_processes)]
        if self.batch_size%num_processes and num_processes > self.batch_size:
            batches.append((self.batch_size%num_processes))
        p = Pool()
        results = p.map(self._make_batch, batches)
        res = mx.stack(results)
        if len(res.shape) == 4:
            res = res.reshape(res.shape[0]*res.shape[1], res.shape[2], res.shape[3])
        p.close()
        p.join()
        return res
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self._start_processed()
