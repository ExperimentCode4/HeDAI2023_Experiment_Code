import pandas as pd

from .RelationConstructor import RelationConstructor
from ..utils.diag_utils import remove_dot, add_dot, roll_to_level
import torch as th

from ..utils import get_midi_df
from ..utils.diag_utils import get_diag_hrchy


class MIMICIVRelationConstructor(RelationConstructor):
    def __init__(self, args, conf):
        super(MIMICIVRelationConstructor, self).__init__(args)
        self.args = args,
        self.conf = conf
        self.path = args.path

    def construct_relations(self):
        # Extract features using feature specific functions
        for relation in self.conf['aux_relations']:
            file_name = relation['data_file']
            sub = relation['sub']
            obj = relation['obj']
            direction = relation['direction']
            func = getattr(self, relation['function'])
            rel = func(file_name, sub, obj, direction)

            # Save the new relation somehow for later consumption
            self.relation_map[(sub, obj)] = rel

    def medi(self, file_name, sub, obj, direction):
        # Medi file contains relations between ingredients and diagnosis codes
        midi_file_path = f'{self.path["data_fold"]}{file_name}.xlsx'

        # drug_ingredient is a bap between drugs and ingredients
        med_map = pd.read_csv(f'{self.path["data_fold"]}drug_ingredient_map.csv', sep=',', dtype=str)

        # Also load the diagnosis hierarchy
        diag_hrchy = get_diag_hrchy(f'{self.path["data_fold"]}diag_subset_hrchy.csv')

        # Also create mapping between levels of the
        med_ing_map = pd.read_excel(midi_file_path, sheet_name='med-ind', dtype='str')

        # Filter for ICD9 relations
        med_ing_map = med_ing_map[med_ing_map.VOCABULARY == 'ICD9CM']

        relation_pairs = []
        for _, row in med_ing_map.iterrows():
            # Find diagnosis code and medication code
            diag_code = diag_hrchy.get_code(remove_dot(row['CODE']))
            ing_code = row['RXCUI']

            # If we do not have the diag code, we continue
            if not diag_code:
                continue

            # map diagnosis code to correct level
            diag_code = roll_to_level(diag_code, self.conf['general']['diag_level'])

            # Map ing_code to medication
            clinical_drugs = med_map[med_map['ingredient'] == ing_code].clinical_drug_form.to_numpy()

            # Since multiple drugs can have the same ingredient
            # We map the diagnosis code to each drug code
            for clinical_drug in clinical_drugs:
                relation_pairs.append([clinical_drug, diag_code.code])

        return relation_pairs, direction
