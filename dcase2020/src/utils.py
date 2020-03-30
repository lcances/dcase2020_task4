from multiprocessing import Process, Manager
import logging
import datetime
import random
import numpy as np
import torch
import logging
import time

# TODO write q timer decorator that deppend on the logging level
def timeit_logging(func):
    def decorator(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        logging.info("%s executed in: %.3fs" % (func.__name__, time.time()-start_time))
        
    return decorator

def feature_cache(func):
    """
    Decorator for the feature extract function. store in the memory the feature calculated, return them if call more
    than once
    IT IS NOT PROCESS / THREAD SAFE.
    Running it into multiple process will result on as mush independant cache than number of process
    """
    def decorator(*args, **kwargs):
        if "filename" in kwargs.keys() and "cached" in kwargs.keys():
            filename = kwargs["filename"]
            cached = kwargs["cached"]

            if filename is not None and cached:
                if filename not in decorator.cache.keys():
                    decorator.cache[filename] = func(*args, **kwargs)
                    return decorator.cache[filename]

                else:
                    if decorator.cache[filename] is None:
                        decorator.cache[filename] = func(*args, **kwargs)
                        return decorator.cache[filename]
                    else:
                        return decorator.cache[filename]

        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator


def multiprocess_feature_cache(func):
    """
    Decorator for the feature extraction function. Perform extraction is not already safe in memory then save it in
    memory. when call again, return feature store in memory
    THIS ONE IS PROCESS / THREAD SAFE
    """
    def decorator(*args, **kwargs):
        if "filename" in kwargs.keys() and "cached" in kwargs.keys():
            filename = kwargs["filename"]
            cached = kwargs["cached"]

            if filename is not None and cached:
                if filename not in decorator.cache.keys():
                    decorator.cache[filename] = func(*args, **kwargs)
                    return decorator.cache[filename]

                else:
                    if decorator.cache[filename] is None:
                        decorator.cache[filename] = func(*args, **kwargs)
                        return decorator.cache[filename]
                    else:
                        return decorator.cache[filename]

        return func(*args, **kwargs)

    decorator.manager = Manager()
    decorator.cache = decorator.manager.dict()

    return decorator


def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]


def get_model_from_name(model_name):
    import models
    import inspect

    for name, obj in inspect.getmembers(models):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_name)
                return obj
    raise AttributeError("This model does not exist: %s " % model_name)


def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False


def set_logs(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)