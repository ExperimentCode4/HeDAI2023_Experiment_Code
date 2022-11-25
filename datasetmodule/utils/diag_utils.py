from datasetmodule.utils.GeneralHierarchy import GeneralHierarchy
import pandas as pd
import torch as th
import json


def create_layered_relations(all_codes, icd_index):
    sou_idx, tar_idx, sib_sou, sib_tar = [], [], [], []

    for code in all_codes:

        # Find parent relation, if any
        parent = code.parent
        if parent is not None:
            sou_idx.append(icd_index[remove_dot(code.code)])
            tar_idx.append(icd_index[remove_dot(parent.code)])

        # Find sibling relations
        for sibling in code.siblings:
            if remove_dot(sibling.code) in icd_index and sibling.code != code.code:
                sib_sou.append(icd_index[remove_dot(code.code)])
                sib_tar.append(icd_index[remove_dot(sibling.code)])

    return sou_idx, tar_idx, sib_sou, sib_tar


def get_icd9_hierarchy_relations(icd_index, icd9_hierarchy):
    # Find unique icd_codes from index
    icd_codes = list(set(icd_index.values()))

    # Create reverse icd_code index
    icd_index_rev = {value: key for key, value in icd_index.items()}

    # Find max code key for subsequent indexing
    max_code_key = max(icd_codes)

    # Find set of all codes + hierarchy codes
    all_codes = set()
    for code in icd_codes:
        current_code = icd9_hierarchy.get_code(add_dot(icd_index_rev[code]))
        all_codes.add(current_code)
        while current_code.parent is not None:
            current_code = current_code.parent
            all_codes.add(current_code)

    # All codes to icd_index with new max_code_indexes
    for code in all_codes:
        code = remove_dot(code.code)
        if code not in icd_index:
            max_code_key += 1
            icd_index[code] = max_code_key
            icd_index_rev[max_code_key] = code

    # Now we create new relation links between all layers
    sou_idx, tar_idx, sib_sou, sib_tar = create_layered_relations(all_codes, icd_index)

    return sou_idx, tar_idx, sib_sou, sib_tar


def roll_to_level(node, level):
    while node.depth > level:
        node = node.parent
    return node


def rollup_code(code, hierarchy, target_level):
    node = hierarchy.get_code(code)
    while node.depth > target_level:
        node = node.parent
    return node.code


def apply_rollup(code, icd9_hierarchy,  level):
    node = icd9_hierarchy.get_code(code)
    node = roll_to_level(node, level)
    return node.code


def remove_dash(code):
    index = code.find('-')
    return code[index + 1:]


def remove_dot(code):
    return code.replace('.', '')


def add_dot(mimic_code):
    if mimic_code is None:
        return None
    if '-' in mimic_code:
        return mimic_code

    icd9_code = ""
    if len(mimic_code) == 3:
        icd9_code = mimic_code
    elif len(mimic_code) > 3:
        mimic_code = list(mimic_code)
        mimic_code.insert(3, ".")
        icd9_code = ''.join(mimic_code)

    return icd9_code


def get_diag_hrchy(file_path):
    hrchy = GeneralHierarchy(file_path)
    return hrchy


def get_used_diagnosis_codes():
    # Read used icd9 codes from file
    file_path = '../data/mimiciv/'
    codes = []
    with open(f'{file_path}/used_diag_codes.csv', 'r') as f:
        for code in f:
            codes.append(code.strip())

    return codes


def prepare_diagnosis_hierarchy_subset():
    # Load the used diagnosis codes
    used_codes = get_used_diagnosis_codes()

    # Read the full ICD9 diagnosis code hierarch
    hierarchy_path = '../data/mimiciv'
    with open(f'{hierarchy_path}/diag_full_hrchy.json', 'r') as f:
        allcodes = json.loads(f.read())
        
        ancestry = []
        found_codes = []
        for hierarchy in allcodes:
            # If no codes from this hierarchy is present in the list of codes
            # used within the MIMIC-IV dataset, we skip the hierarchy path
            if not {remove_dot(node['code']) for node in hierarchy}.intersection(set(used_codes)):
                continue

            # Here we know the code is important for us, so we store its ancestry
            for i, (child, parent) in enumerate(zip(hierarchy[1:], hierarchy[0:-1])):
                # Add 'ROOT' if first relation
                if i == 0:
                    ancestry.append([remove_dot(parent['code']), 'ROOT', parent['descr']])

                ancestry.append([remove_dot(child['code']), remove_dot(parent['code']), child['descr']])

            found_codes.append(remove_dot(child['code']))

        # Convert the ancestry to a pandas dataframe
        subset_hierarchy = pd.DataFrame(ancestry, columns=['CHILD', 'PARENT', 'DESCR'])

        # Drop duplicates and save hierarchy
        subset_hierarchy.drop_duplicates(inplace=True)
        subset_hierarchy.to_csv(f'{hierarchy_path}/hrchy_subset_diag.csv', sep=';', index=False)


def add_icd9_hiearchy_relations(self, sub, obj, rev_edges=True, add_siblings=False):
    # Find their relations
    sou_idx, tar_idx, sib_sou, sib_tar = get_icd9_hierarchy_relations(self.indexes[sub], self.icd9_hierarchy)
    sources, targets = th.tensor(sou_idx, dtype=th.int64), th.tensor(tar_idx, dtype=th.int64)
    self.relations_set[(sub, f'{sub}-{obj}', obj)] = (sources, targets)

    if rev_edges:
        self.relations_set[(obj, f'{obj}-{sub}', sub)] = (targets, sources)

    if add_siblings:
        # TODO: The relation name has to change for this to work
        sources, targets = th.tensor(sib_sou, dtype=th.int64), th.tensor(sib_tar, dtype=th.int64)
        self.relations_set[(sub, f'{sub}-s-{obj}', obj)] = (targets, sources)
        print(f'Finished tuple ({sub}, {sub}-s-{obj}, {obj}) with {len(sib_sou)} relations between '
              f'{len(sib_sou)} {sub} and {len(sib_tar)} {obj}')

    print(f'Finished tuple ({sub}, {sub}-{obj}, {obj}) with {len(sou_idx)} relations between '
          f'{len(sou_idx)} {sub} and {len(tar_idx)} {obj}')


def get_rollup_map(init_map: dict, out_depth: int, diag_hrchy: GeneralHierarchy) -> dict:
    """
    Parameters
    ----------
    init_map: The initial map current label indexes to corresponding ICD9 codes
    out_depth: The depth we with to roll to
    icd9_hrchy: ICD9 hierarchy

    Returns:
    -------
    out_map: tensor mappings from initial column indexes to output column indexes
    """
    # New mapping between ICD9 codes and their label indexes
    new_map = {}

    for key, value in init_map.items():
        node = diag_hrchy.get_code(value)

        # Find the corresponding node of specific depth
        while node.depth > out_depth:
            node = node.parent

        # Add the node to the new map
        if node.code in new_map:
            new_map[node.code].add(key)
        else:
            new_map[node.code] = {key}

    # Create map between indexes of initial tensor, and rolled up tensor
    out_map = {index: list(columns) for index, columns in enumerate(new_map.values())}

    return out_map


def rollup_tensor(input_tensor: th.Tensor, aggregation: dict, agg_func: callable) -> th.tensor:

    agg_tensor = th.zeros(size=(input_tensor.shape[0], len(aggregation)), dtype=th.float32)
    for i, agg_indexes in enumerate(aggregation.values()):
        agg_tensor[:, i] = agg_func(input_tensor[:, agg_indexes], dim=1)[0]

    return agg_tensor
