import functools
import inspect


def DEBUG_WRAPPER(func):
    '''This is a decorator that will wrap a function and print out the args and kwargs of the function if the DEBUG flag is set to true
    ______
    takes:
    func (function): the function to wrap
    returns:
    wrapper (function): the wrapped function
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'DEBUG' in globals():
            if globals()['DEBUG']:
                print(f"=== Entering {func.__name__} ===")
                print(f"args: {args}")
                print(f"kwargs: {kwargs}")
                return func(*args, **kwargs)
            else:
                #if not in debug mode, we want to inspect the function and see its return type
                #if the return type is a dataframe, we want to return a copy of the dataframe
                #if the return type is a list, we want to return a copy of the list etc
                return_type = inspect.signature(func).return_annotation
                #try to run the function, if it fails, we want to return the empty value
                try:
                    func(*args, **kwargs)
                except:
                    #if we get here, we have not implemented the return type
                    print(f"Error: {func.__name__} has return type {return_type} and raised an error")
                    return return_type()
                if return_type == inspect._empty:
                    return 
                else:
                    
                    raise NotImplementedError(f"Return type {return_type} not implemented")

        else:
            #if we get here the user has not set the debug flag, so we will just run the function, or deleted the debug flag?
            return func(*args, **kwargs)
            

    return wrapper