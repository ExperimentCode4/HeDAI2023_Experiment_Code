import json

import pandas as pd
import torch as th
import numpy as np
import random
import dgl

from .diag_utils import roll_to_level


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    dgl.seed(seed)


def get_midi_df(midi_file_path, icd9_hierarchy):
    # Load xlsx file
    df = pd.read_excel(midi_file_path, sheet_name='med-ind', dtype='str')

    # Filter for ICD9 relations
    df = df[df.VOCABULARY == 'ICD9CM']

    l5_list, l4_list, l3_list, l2_list, l1_list = [], [], [], [], []
    codes_not_found = []

    # For each code, find correponsing rolled up codes
    for init_code in df['CODE']:
        init_code = init_code.replace(".", '')

        # Ranges of codes are too unspecific
        if '-' in init_code:
            codes_not_found.append(init_code)
            continue

        code = icd9_hierarchy.get_code(init_code)
        if not code:
            codes_not_found.append(init_code)
            continue

        l5 = roll_to_level(code, 5)
        l4 = roll_to_level(l5, 4)
        l3 = roll_to_level(l4, 3)
        l2 = roll_to_level(l3, 2)
        l1 = roll_to_level(l2, 1)

        l5_list.append(code.code)
        l4_list.append(l4.code)
        l3_list.append(l3.code)
        l2_list.append(l2.code)
        l1_list.append(l1.code)

    df = df[~df.CODE.isin(codes_not_found)]

    df.reindex()
    df['level5'] = l5_list
    df['level4'] = l4_list
    df['level3'] = l3_list
    df['level2'] = l2_list
    df['level1'] = l1_list

    return df


def get_rxnorm_atc_map(file_path):
    rxnorm_to_atc = []
    with open(file_path, 'r') as f:
        # allcodes are all the level 5 atc codes
        atc_manual_map = json.loads(f.read())
        for row in atc_manual_map:
            rxnorm_to_atc.append([row['concept_code'], row['atc_code']])
    return rxnorm_to_atc
