import yaml
from pprint import pprint

from flow_merge.lib.loaders.normalizer import NormalizationRunner


def test_template():
    yaml_input = """
    """
    expected = []

    yaml_loaded = yaml.safe_load(yaml_input)
    pprint(yaml_loaded)
    pprint(type(yaml_loaded))
    processed = NormalizationRunner("qwen1_5.json").normalize([yaml_loaded])
    pprint(processed)
    assert processed == expected
