import re
import warnings
from typing import Optional


def parse_cre_line(full_genotype: str) -> Optional[str]:
    """
    Parameters
    ----------
    full_genotype
        formatted from LIMS, e.g.
        Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt

    Returns
    ----------
    cre_line
        just the Cre line, e.g. Vip-IRES-Cre, or None if not possible to parse
    """
    if ';' not in full_genotype:
        return None
    return full_genotype.split(';')[0].replace('/wt', '')


def parse_age_in_days(age: str) -> Optional[int]:
    """Converts the age code into a numeric days representation

    Parameters
    ----------
    age
        age code, ie P123
    """
    if not age.startswith('P'):
        warnings.warn('Could not parse numeric age from age code')
        return None

    match = re.search(r'\d+', age)

    if match is None:
        warnings.warn('Could not parse numeric age from age code')
        return None

    start, end = match.span()
    return int(age[start:end])
