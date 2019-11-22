"""BLM name to type mapping.
"""
import re

TYPE_MAP = {'coll': [r'_T[C|D]'],
            'xrp': [r'_XRP'],
            'cold': [r'_MQM[L]?',
                     r'_MQ(-.*)?$',
                     r'_MQY',
                     r'_MQTL',
                     r'_MQX[AB]$',
                     r'_MB[AB]',
                     r'_MBR[BCS]',
                     r'_MBX[AB]',
                     r'_LE[ABCDEFGHIJ][LR]',
                     r'_ACSGA',
                     r'_BKGD',
                     r'_DFBLB'],
            'warm': [r'_MQW[AB]',
                     r'_TANC',
                     r'_TANA[RL]',
                     r'_MSI[AB]',
                     r'_MSD[ABC]',
                     r'_MK[ID]',
                     r'_MBXW',
                     r'_BSRT',
                     r'_BPMSW',
                     r'_MBW',
                     r'_BGI',
                     r'_X5FC[AB]']}


def name_to_type(name, on_fail='other'):
    """Converts find the type corresponding a given BLM name.

    Args:
        name (str): BLM name.
        on_fail (str, optional): returned if not BLM type is found.

    Returns:
        str: the BLM type or "on_fail" if no BLM type is found.
    """

    for k, regex_l in TYPE_MAP.items():
        res = [re.search(reg, name) for reg in regex_l]
        if any(res):
            return k
    return on_fail
