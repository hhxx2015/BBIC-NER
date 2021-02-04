

from __future__ import absolute_import,division,print_function
import multiprocessing
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process


import zmq
from termcolor import colored

from .helper import *
sys.path.append('..')

__all__ = ['__version__', 'BertServer']
__version__ = '1.7.8'


def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    assert int(tf_ver[0]) >= 1 and int(tf_ver[1]) >= 10, 'Tensorflow >=1.10 is required!'
    return tf_ver

_tf_ver_ = check_tf_version()
 

class ServerCommand:
    
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'



class BertServer(threading.Thread):
    
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port
        self.args = args
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'tensorflow_version': _tf_ver_,
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.processes = []


class BertSink(Process):
    
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'), args.verbose)
        self.front_sink_addr = front_sink_addr
        self.verbose = args.verbose
        self.args = args



class BertWorker(Process):
    
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, mode, id2label=None):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
        self.max_seq_len = args.max_seq_len
        self.mask_cls_sep = args.mask_cls_sep
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.verbose = args.verbose
        self.graph_path = graph_path
        self.use_fp16 = args.fp16
        self.args = args
        self.mode = mode
        self.id2label = id2label


class ServerStatistic:
 
    def __init__(self):
        self._hist_client = defaultdict(int)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = defaultdict(float)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        
        client, msg, req_id, msg_len = request
        self._hist_client[client] += 1
        if ServerCommand.is_valid(msg):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp
