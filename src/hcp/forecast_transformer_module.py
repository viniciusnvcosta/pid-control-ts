# ABOUTME: Retains the registered Darts Transformer initialization and Adam identity.
# ABOUTME: Keeps architecture construction separate from selection and forecast orchestration.
"""Unchanged Transformer architecture used by both lifecycle characterization paths."""

from darts.models import TransformerModel
from darts.models.forecasting.transformer_model import _TransformerModule


class _PersistentTransformerModule(_TransformerModule):
    """Darts Transformer module that retains one Adam state over epoch calls."""

    def configure_optimizers(self):
        if not hasattr(self, "_persistent_optimizer"):
            self._persistent_optimizer = super().configure_optimizers()
        return self._persistent_optimizer


class _PersistentTransformer(TransformerModel):
    """Darts Transformer with the same model and a stateful Adam factory."""

    def _create_model(self, train_sample):
        module = super()._create_model(train_sample)
        persistent = _PersistentTransformerModule(
            input_size=module.input_size,
            output_size=module.target_size,
            nr_params=module.nr_params,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            norm_type=self.norm_type,
            custom_encoder=self.custom_encoder,
            custom_decoder=self.custom_decoder,
            **self.pl_module_params,
        )
        persistent.load_state_dict(module.state_dict())
        return persistent
