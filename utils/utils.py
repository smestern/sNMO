import functools
DEBUG= None
def DEBUG_WRAPPER(func):
    def wrapper(*args, **kwargs):
        if DEBUG in globals():
            if DEBUG:
                print(f"=== Entering {func.__name__} ===")
                print(f"args: {args}")
                print(f"kwargs: {kwargs}")
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper