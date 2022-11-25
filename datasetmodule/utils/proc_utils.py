from collections import namedtuple
import pandas as pd
import itertools
import os

from datasetmodule.utils.GeneralHierarchy import GeneralHierarchy


def get_used_procedure_codes():
    # Read used loinc codes from file
    file_path = os.path.join(os.getcwd(), '../data/mimiciv/')

    codes = []
    with open(f'{file_path}used_proc_codes.csv', 'r') as f:
        for code in f:
            codes.append(code.strip())

    return codes


# I could not find the root of the ICD-9-Procedure codes online 
# so I compiled the first level manually, as this is the only level
# where there is no clear rule for how to group children
def get_initial_level():
    codes = []
    code_level = namedtuple('CodeLevel', 'start end descr code')
    # List of top level codes...
    codes.append(code_level(0, 0, 'PROCEDURES AND INTERVENTIONS , NOT ELSEWHERE CLASSIFIED', '00-00'))
    codes.append(code_level(1, 5, 'OPERATIONS ON THE NERVOUS SYSTEM', '01-05'))
    codes.append(code_level(6, 7, 'OPERATIONS ON THE ENDOCRINE SYSTEM', '06-07'))
    codes.append(code_level(8, 16, 'OPERATIONS ON THE EYE', '08-16'))
    codes.append(code_level(17, 17, 'OTHER MISCELLANEOUS DIAGNOSTIC AND THERAPEUTIC PROCEDURES', '17-17'))
    codes.append(code_level(18, 20, 'OPERATIONS ON THE EAR', '18-20'))
    codes.append(code_level(21, 29, 'OPERATIONS ON THE NOSE, MOUTH, AND PHARYNX', '21-29'))
    codes.append(code_level(30, 34, 'OPERATIONS ON THE RESPIRATORY SYSTEM', '30-34'))
    codes.append(code_level(35, 39, 'OPERATIONS ON THE CARDIOVASCULAR SYSTEM', '35-39'))
    codes.append(code_level(40, 41, 'OPERATIONS ON THE HEMIC AND LYMPHATIC SYSTEM', '40-41'))
    codes.append(code_level(42, 54, 'OPERATIONS ON THE DIGESTIVE SYSTEM', '42-54'))
    codes.append(code_level(55, 59, 'OPERATIONS ON THE URINARY SYSTEM', '55-59'))
    codes.append(code_level(60, 64, 'OPERATIONS ON THE MALE GENITAL ORGANS', '60-64'))
    codes.append(code_level(65, 71, 'OPERATIONS ON THE FEMALE GENITAL ORGANS', '65-71'))
    codes.append(code_level(72, 75, 'OBSTETRICAL PROCEDURES', '72-75'))
    codes.append(code_level(76, 84, 'OPERATIONS ON THE MUSCULOSKELETAL SYSTEM', '76-84'))
    codes.append(code_level(85, 86, 'OPERATIONS ON THE INTEGUMENTARY SYSTEM', '85-86'))
    codes.append(code_level(87, 99, 'MISCELLANEOUS DIAGNOSTIC AND THERAPEUTIC PROCEDURES', '87-99'))

    return codes


def prepare_procedures_subset():
    # Read the procedure codes used in MIMIC-IV
    used_codes = get_used_procedure_codes()

    # Get level 1 codes
    l1_codes = get_initial_level()

    data_rows = []
    for code in used_codes:
        depth = get_code_depth(code)

        # If depth is three, we find it's parent
        if depth == 3:
            data_rows.append([code, code[0:-1]])
            code = code[0:-1]

        # Now the depth is 2, so we connect to a l1 parent code
        for l1_code in l1_codes:
            if int(code[0:-2]) in range(l1_code.start, l1_code.end + 1):
                data_rows.append([code, l1_code.code])
                data_rows.append([l1_code.code, 'ROOT'])
                break

    # Now we remove duplicates
    data_rows.sort()
    data_rows = list(k for k, _ in itertools.groupby(data_rows))

    # Create pandas dataframe and save as csv for further processing
    file_path = os.path.join(os.getcwd(), '../data/mimiciv')
    df = pd.DataFrame(data_rows, columns=['CHILD', 'PARENT'])
    df.to_csv(f'{file_path}/hrchy_subset_proc.csv', sep=';', index=False)


# Find the depth of a code, for ICD-9 procedures
# There are only three levels and the ROOT node
def get_code_depth(code):
    # Should only be used for leaf nodes
    code = code.replace('.', '')
    if len(code) == 4:
        return 3
    if len(code) == 3:
        return 2
    else:
        return None


def get_proc_hrchy(code_path, index_leaf=False):
    hrchy = GeneralHierarchy(code_path, index_leaf)
    return hrchy
