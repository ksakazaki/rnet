import random
import string


def random_string(prepend='', append='', length=20):
    '''
    Return random string of given length.

    Parameters
    ----------
    prepend, append : str, optional
        String to prepend or append to the random string. The defaults
        are empty strings.
    length : int, optional
        String length. The default is 20.

    Returns
    -------
    str
    '''
    letters = string.ascii_lowercase + string.digits
    return prepend + ''.join(random.choice(letters) for _ in range(length)) + append
