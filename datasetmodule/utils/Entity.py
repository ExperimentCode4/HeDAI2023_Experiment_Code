from pandas import DataFrame


class Entity:
    def __init__(self,
                 name: str,
                 alias: str):
        self.name = name
        self.alias = alias
        self.ids = set()
        self.maps = dict()
        self.filter = dict()
        self.feat_map = dict()

    def populate(self, df: DataFrame):
        self.ids = set(df[self.name].unique())

    def reindex(self, new_ids: set):
        # Save only the ids being used in relations
        self.ids = new_ids.intersection(self.ids)

        # Re-index the remaining ids and add to map
        reindex_origin_map = {new_id: old_id for new_id, old_id in enumerate(self.ids)}
        origin_reindex_map = {old_id: new_id for new_id, old_id in reindex_origin_map.items()}

        # Add maps to class for backwards compatability
        self.maps['origin-reindex'] = origin_reindex_map
        self.maps['reindex-origin'] = reindex_origin_map

        # Map ids to re-indexed ids
        self.ids = set([self.maps['origin-reindex'][id] for id in self.ids])
