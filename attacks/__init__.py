import importlib

_method_mapping = {
    "GCG": "attacks.gcg", 
    "MGCG_ST": "attacks.multi_gcg_same_tokenizer",
    "MGCG_DT": "attacks.multi_gcg_different_tokenizer",
}

def get_method_class(method):
    """
    Returns the method class given the method name. This is used to access static methods.
    """
    if method not in _method_mapping:
        raise ValueError(f"Can not find method {method}")
    
    module_path = _method_mapping[method]
    
    try:
        module = importlib.import_module(module_path)  # Import the module (not class)
    except ModuleNotFoundError as e:
        raise ImportError(f"Failed to import {module_path}: {e}")

    method_class = getattr(module, method)  # Get the class from the module
    return method_class
