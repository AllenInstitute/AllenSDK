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


def parse_age_in_days(age: str):
    """Converts the age string into a numeric days representation

    Parameters
    ----------
    age
        String representation of age (ie P123)
    """
    if len(age) == 0:
        return None

    if age[0] != 'P':
        return None
    try:
        age = int(age[1:])
        return age
    except ValueError:
        return None
