from datasetmodule.utils.GeneralHierarchy import GeneralHierarchy
import pandas as pd


def get_used_loinc_lab_codes():
    # Read used loinc codes from file
    file_path = '../data/mimiciv/'

    codes = set()
    with open(f'{file_path}/used_loinc_lab_codes.csv', 'r') as f:
        for code in f:
            codes.add(code.strip())

    return codes


def prepare_loinc_hierarchy_subset():
    # Read the full loinc axial hierarch
    hierarchy_path = '../data/mimiciv'
    full_hierarchy = pd.read_excel(f'{hierarchy_path}/full_hrchy_loinc.xlsx')

    # Read the loinc codes used in MIMIC-IV
    used_codes = get_used_loinc_lab_codes()

    # Find subset of rows from full hierarch
    # where only the used_codes are included
    subset_hierarchy = full_hierarchy[full_hierarchy['CODE'].isin(used_codes)]

    # We have some duplicates root to child paths resulting in multiple
    # parents for some concepts. For now we only keep the first concept
    subset_hierarchy = subset_hierarchy.drop_duplicates(subset='CODE')

    # Iterate though the rows of the dataframe
    # to create new pairs of parents and children
    pairs = []
    parents = []

    for _, row in subset_hierarchy.iterrows():
        code = row['CODE']
        root_path = row['PATH_TO_ROOT'].split('.')

        # Add root to the root_path_nodes
        root_path.insert(0, 'ROOT')
        root_path.append(code)

        root_path.reverse()

        # Create pairs of codes
        for child, parent in zip(root_path[:-1], root_path[1:]):
            pairs.append([child, parent])
            if parent in parents:
                break
            else:
                parents.append(parent)

    subset_hrchy = pd.DataFrame(data=pairs, columns=['CHILD', 'PARENT'])
    subset_hrchy.drop_duplicates(inplace=True)
    # As we are not too interested in the text of the intermediary
    # nodes of the hierarchy right now, we simply save the new
    # subset_hierarchy to a new file for later processing
    subset_hrchy.to_csv(f'{hierarchy_path}/hrchy_subset_loinc.csv', sep=';', index=False)


def get_loinc_hrchy(code_path, index_leaf=False):
    hrchy = GeneralHierarchy(code_path, index_leaf)
    return hrchy
