import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from flow_merge.lib.planners.planner import Planner
from flow_merge.lib.snapshot.data_architecture.snapshot import Snapshot
from flow_merge.lib.snapshot.data_architecture._metadata import SnapshotMetadata, SnapshotHost
from flow_merge.lib.snapshot.data_architecture._settings import MergeSettings
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSlices, NormalizedSlice, NormalizedSource
from flow_merge.lib.model.model import Model
from flow_merge.lib.tokenizer import Tokenizer
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.logger import Logger
from flow_merge.lib.enriched_snapshot import EnrichedSnapshot
from flow_merge.lib.validators._directory_settings import DirectorySettings
from flow_merge.lib.validators._model_settings import ModelSettings, RawModelDict
from flow_merge.lib.validators._method_settings import MethodSettings, MergeMethodIdentifier
from flow_merge.lib.validators._tokenizer_settings import TokenizerSettings

class TestPlanner(unittest.TestCase):

    def setUp(self):
        # Mocking the environment and logger
        self.env = MagicMock(spec=ApplicationConfig)
        self.logger = MagicMock(spec=Logger)

        # Creating a mock snapshot object
        self.snapshot_metadata = SnapshotMetadata(
            created_at=datetime.now().isoformat(),
            library_version="1.0.0",
            host=SnapshotHost(os="Linux", system_architecture="x86_64"),
            sha="metadata_sha"
        )
        self.directory_settings = DirectorySettings()

        # Initializing ModelSettings
        self.model_settings = ModelSettings(
            base_model="base_model_path",
            models=[
                RawModelDict(model="base_model_path"),
                RawModelDict(model="model_2")
            ]
        )

        # Initializing MethodSettings
        self.method_settings = MethodSettings(
            method=MergeMethodIdentifier.PASSTHROUGH
        )

        # Initializing TokenizerSettings
        self.tokenizer_settings = TokenizerSettings(
            mode="base",
            interpolation_method="linear"
        )

        # Initializing MergeSettings with all required fields
        self.merge_settings = MergeSettings(
            directory_settings=self.directory_settings,
            models=self.model_settings,
            method=self.method_settings,
            tokenizer=self.tokenizer_settings,
            sha="settings_sha"
        )

        # Initializing NormalizedSlices with required fields
        self.normalized_slices = [
                NormalizedSlice(
                    index=0,
                    merge_method=MergeMethodIdentifier.PASSTHROUGH,
                    sources=[NormalizedSource(base_model=True, layer="model.embed_tokens.weight", model="model_1")]
                ),
                NormalizedSlice(
                    index=7,
                    merge_method=MergeMethodIdentifier.PASSTHROUGH,
                    sources=[
                        NormalizedSource(base_model=True, layer="model.layers.0.mlp.gate_proj.weight", model="model_1"),
                        NormalizedSource(base_model=False, layer="model.layers.0.mlp.gate_proj.weight", model="model_2")
                    ]
                ),
                NormalizedSlice(
                    index=28,
                    merge_method=MergeMethodIdentifier.PASSTHROUGH,
                    sources=[
                        NormalizedSource(base_model=True, layer="model.norm.weight", model="model_1"),
                        NormalizedSource(base_model=False, layer="model.norm.weight", model="model_2")
                    ]
                )
            ]
        self.snapshot = Snapshot(
            sha='test_sha', 
            metadata=self.snapshot_metadata, 
            settings=self.merge_settings, 
            normalized=self.normalized_slices
        )

        # Mocking Model class
        self.model_class = MagicMock(spec=Model)

        # Creating an instance of the Planner
        self.planner = Planner(env=self.env, logger=self.logger, snapshot=self.snapshot, model_class=self.model_class)

    @patch('flow_merge.lib.planners.planner.extract_models_by_layers')
    @patch('flow_merge.lib.planners.planner.EnrichedSnapshot')
    @patch('flow_merge.lib.planners.planner.MergeTokenizerService')
    def test_plan(self, mock_merge_tokenizer_service, mock_enriched_snapshot, mock_extract_models_by_layers):
        # Mocking the return values of extract_models_by_layers
        mock_extract_models_by_layers.return_value = MagicMock()
        mock_extract_models_by_layers.return_value.base_model = 'base_model_path'
        mock_extract_models_by_layers.return_value.models = {'model_1': 'layers_1', 'model_2': 'layers_2'}

        # Mocking Model class methods
        base_model = MagicMock(spec=Model)
        model_1 = MagicMock(spec=Model)
        model_2 = MagicMock(spec=Model)
        self.model_class.from_path.return_value = base_model
        self.model_class.from_layers.side_effect = [model_1, model_2]

        # Mocking EnrichedSnapshot
        mock_enriched_snapshot.return_value = MagicMock(spec=EnrichedSnapshot)

        # Mocking MergeTokenizerService
        mock_merge_tokenizer_service.get_merge_tokenizer.return_value = MagicMock(spec=Tokenizer)

        # Running the plan method
        enriched_snapshot = self.planner.plan()

        # Asserting the calls and results
        self.model_class.from_path.assert_called_once_with(
            path='base_model_path',
            directory_settings=self.directory_settings,
            env=self.env,
            logger=self.logger
        )

        self.model_class.from_layers.assert_any_call(
            layers_to_download='layers_1',
            path='model_1',
            directory_settings=self.directory_settings,
            env=self.env,
            logger=self.logger
        )

        self.model_class.from_layers.assert_any_call(
            layers_to_download='layers_2',
            path='model_2',
            directory_settings=self.directory_settings,
            env=self.env,
            logger=self.logger
        )

        mock_merge_tokenizer_service.get_merge_tokenizer.assert_called_once_with(
            mock_enriched_snapshot.return_value, self.env, self.logger
        )

        self.assertTrue(hasattr(enriched_snapshot, 'tokenizer'))

if __name__ == '__main__':
    unittest.main()
