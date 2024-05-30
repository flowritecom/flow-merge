from typing import Optional, Dict, Any

from flow_merge.lib.legality_check._first_check import FirstCheck


class LegalityCheckRunner:
    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data
        self.first_check = self.check(FirstCheck, ['first_check'])

    def check(self, check_class, keys):
        return check_class(
            **{k: self.raw_data[k] for k in keys if k in self.raw_data}
        )
