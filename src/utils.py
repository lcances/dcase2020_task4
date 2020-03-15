from multiprocessing import Process, Manager


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
