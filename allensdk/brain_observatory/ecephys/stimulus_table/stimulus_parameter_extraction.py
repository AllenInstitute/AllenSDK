import re
import ast


REPR_PARAMS_RE = re.compile(r'([a-z0-9]+=[^=]+)[,\)]', re.IGNORECASE)
REPR_CLASS_RE = re.compile(r'^(?P<class_name>[a-z0-9]+)\(.*\)$', re.IGNORECASE)
ARRAY_RE = re.compile(r'array\((?P<contents>\[.*\])\)')

DROP_PARAMS = ( # psychopy boilerplate, more or less
    'name', 
    'autoLog',
    'autoDraw',
    'win'
)

def extract_stim_repr(
    stim_repr, drop_params=DROP_PARAMS,
    repr_class_re=REPR_CLASS_RE, repr_params_re=REPR_PARAMS_RE, array_re=ARRAY_RE
):

    stim_class = extract_stim_class_from_repr(stim_repr, class_re=repr_class_re)
    stim_params = extract_const_params_from_stim_repr(stim_repr, repr_params_re=repr_params_re, array_re=array_re)

    for drop_param in drop_params:
        if drop_param in stim_params:
            del stim_params[drop_param]

    if stim_class is not None and len(stim_params) > 0:
        
        if stim_class == 'DotStim':
            return extract_dot_stim_const_params(stim_params)
        elif stim_class == 'GratingStim':
            return extract_grating_stim_const_params(stim_params)
        else:
            raise ValueError(f'unrecognized stimulus class: {stim_class}')


def extract_stim_class_from_repr(stim_repr, repr_class_re=REPR_CLASS_RE):
    match = repr_class_re.match(stim_repr)
    if match is not None and 'class_name' in match:
        return match['class_name']


def extract_const_params_from_stim_repr(stim_repr, repr_params_re=REPR_PARAMS_RE, array_re=ARRAY_RE):
    '''Parameters which are not set as sweep_params in the stimulus script (usually because they are not 
    varied during the course of the session) are not output in an easily machine-readable format. This function 
    attempts to recover them by parsing the string repr of the stimulus.

    Parameters
    ----------
        stim_repr : str
            The repr of the camstim stimulus object. Served up per-stimulus in the stim pickle.
        repr_params_re : re.Pattern
            Extracts attributes as "="-seperated strings
        array_re : re.Pattern
            Extracts list reprs from numpy array reprs.

    Returns
    -------
    repr_params : dict
        dictionary of paramater keys and values extracted from the stim repr. Where possible, the values are converted 
        to native Python types.

    '''

    repr_params = {}

    for match in repr_params_re.findall(stim_repr):
        k, v = match.split('=')
        
        if k not in repr_params:

            m = array_re.match(v)
            if m is not None:
                v = m['contents']

            try:
                v = ast.literal_eval(v)
            except ValueError as err:
                pass

            repr_params[k] = v

        else:
            raise KeyError(f'duplicate key: {k}')

    return repr_params
