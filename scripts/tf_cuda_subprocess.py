# code from https://gist.github.com/ed-alertedh/85dc3a70d3972e742ca0c4296de7bf00
import os
import cloudpickle
from multiprocessing import Pool

class RunAsCUDASubprocess:
    def __init__(self, num_gpus=0, memory_fraction=0.8):
        self._num_gpus = num_gpus
        self._memory_fraction = memory_fraction

    @staticmethod
    def _subprocess_code(num_gpus, memory_fraction, fn, args):
        # set the env vars inside the subprocess so that we don't alter the parent env
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see tensorflow issue #152
        try:
            import py3nvml
            num_grabbed = py3nvml.grab_gpus(num_gpus, gpu_fraction=memory_fraction)
        except:
            # either CUDA is not installed on the system or py3nvml is not installed (which probably means the env
            # does not have CUDA-enabled packages). Either way, block the visible devices to be sure.
            num_grabbed = 0
            os.environ['CUDA_VISIBLE_DEVICES'] = ""

        assert num_grabbed == num_gpus, 'Could not grab {} GPU devices with {}% memory available'.format(
            num_gpus,
            memory_fraction * 100)
        if os.environ['CUDA_VISIBLE_DEVICES'] == "":
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # see tensorflow issues: #16284, #2175

        # using cloudpickle because it is more flexible about what functions it will
        # pickle (lambda functions, notebook code, etc.)
        return cloudpickle.loads(fn)(*args)

    def __call__(self, f):
        def wrapped_f(*args):
            with Pool(1) as p:
                return p.apply(RunAsCUDASubprocess._subprocess_code, (self._num_gpus, self._memory_fraction, cloudpickle.dumps(f), args))

        return wrapped_f