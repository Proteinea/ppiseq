from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
from transformers.models import convbert
from ppi_research.layers import poolers


class AttnPoolAddConvBERTModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        model_name: str,
        embedding_name: str,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=False,
        )
        self.pooler = poolers.get(pooler)

        convbert_config = convbert.ConvBertConfig(
            hidden_size=self.embed_dim,
            num_hidden_layers=1,
            num_attention_heads=8,
            intermediate_size=self.embed_dim // 2,
            conv_kernel_size=7,
        )

        # We use only one convbert layer in
        # our benchmarking so we just use `ConvBertLayer`.
        self.convbert_layer = convbert.ConvBertLayer(convbert_config)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.regressor = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.regressor.bias.data.zero_()
        self.regressor.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        ligand_input_ids,
        receptor_input_ids,
        ligand_attention_mask,
        receptor_attention_mask,
        labels=None,
    ):
        ligand_embed, receptor_embed = self.backbone(
            ligand_input_ids,
            receptor_input_ids,
            ligand_attention_mask,
            receptor_attention_mask,
        )

        ligand_attention_mask = ligand_attention_mask.to(
            device=ligand_embed.device,
            dtype=ligand_embed.dtype,
        )

        receptor_attention_mask = receptor_attention_mask.to(
            device=receptor_embed.device,
            dtype=receptor_embed.dtype,
        )

        ligand_embed = self.convbert_layer(ligand_embed)[0]
        receptor_embed = self.convbert_layer(receptor_embed)[0]

        output_1, _ = self.attn(
            query=ligand_embed,
            key=receptor_embed,
            value=receptor_embed,
            key_padding_mask=receptor_attention_mask.log(),
            need_weights=False,
        )
        output_2, _ = self.attn(
            query=receptor_embed,
            key=ligand_embed,
            value=ligand_embed,
            key_padding_mask=ligand_attention_mask.log(),
            need_weights=False,
        )

        output_1 = output_1 + ligand_embed
        output_2 = output_2 + receptor_embed
        pooled_output_1 = self.pooler(output_1, ligand_attention_mask)
        pooled_output_2 = self.pooler(output_2, receptor_attention_mask)
        pooled_output = pooled_output_1 + pooled_output_2
        logits = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {"logits": logits, "loss": loss}
