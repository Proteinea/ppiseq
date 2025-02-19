from __future__ import annotations
from torch import nn
import torch
from typing import Dict
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from ppi_research.layers import poolers


def aggregate_chains(sequence_1, sequence_2, aggregation_method: str):
    if aggregation_method == "concat":
        return torch.cat([sequence_1, sequence_2], dim=-1)
    elif aggregation_method == "mean":
        return (sequence_1 + sequence_2) / 2
    elif aggregation_method == "add":
        return sequence_1 + sequence_2
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")


class MultiChainModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        global_pooler: nn.Module | str,
        chains_pooler: nn.Module | str,
        shared_global_pooler: bool = False,
        shared_chains_pooler: bool = False,
        aggregation_method: str = "concat",
        use_ffn: bool = False,
        bias: bool = False,
        model_name: str | None = None,
        embedding_name: str | None = None,
    ):
        super().__init__()

        self.embed_dim = backbone.config.hidden_size
        self.use_ffn = use_ffn
        self.shared_global_pooler = shared_global_pooler
        self.shared_chains_pooler = shared_chains_pooler
        self.aggregation_method = aggregation_method
        input_dim = (
            self.embed_dim * 2
            if self.aggregation_method == "concat"
            else self.embed_dim
        )

        if self.shared_global_pooler:
            self.global_pooler = poolers.get(
                global_pooler, self.embed_dim
            )
        else:
            self.ligand_global_pooler = poolers.get(
                global_pooler, self.embed_dim
            )
            self.receptor_global_pooler = poolers.get(
                global_pooler, self.embed_dim
            )

        if self.shared_chains_pooler:
            self.chains_pooler = poolers.get(
                chains_pooler, self.embed_dim
            )
        else:
            self.ligand_chains_pooler = poolers.get(
                chains_pooler, self.embed_dim
            )
            self.receptor_chains_pooler = poolers.get(
                chains_pooler, self.embed_dim
            )

        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
        )

        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=bias),
                nn.SiLU(),
                nn.Linear(input_dim, input_dim, bias=bias),
            )

        self.output = nn.Linear(input_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def process_chains_v2(
        self,
        protein_embed: torch.FloatTensor,
        chain_ids: torch.LongTensor,
        pooler: nn.Module,
    ) -> torch.FloatTensor:
        """Vectorized implementation of the chains pooling.

        Args:
            protein_embed (torch.FloatTensor): The protein embeddings.
            chain_ids (torch.LongTensor): The chain ids.
            pooler (nn.Module): The pooler to use.

        Returns:
            torch.FloatTensor: The pooled chains.
        """
        # Vectorized implementation
        unique_chains = torch.unique(chain_ids)

        # Create a mask of shape [num_unique_chains, batch_size]
        masks = torch.stack(
            [(chain_ids == chain_id) for chain_id in unique_chains],
        )

        # Expand protein_embed for broadcasting
        expanded_embed = protein_embed.unsqueeze(0).expand(
            len(unique_chains),
            -1,
            -1,
        )

        # Apply masks and pool
        # masked_embeds = expanded_embed * masks.unsqueeze(-1)

        pooled_chains = pooler(expanded_embed, masks)
        return pooled_chains

    def process_chains(
        self,
        protein_embed: torch.FloatTensor,
        chain_ids: torch.LongTensor,
        pooler: nn.Module,
    ) -> torch.FloatTensor:
        """Non-vectorized implementation of the chains pooling.

        Args:
            protein_embed (torch.FloatTensor): The protein embeddings.
            chain_ids (torch.LongTensor): The chain ids.
            pooler (nn.Module): The pooler to use.

        Returns:
            torch.FloatTensor: The pooled chains.
        """
        outputs = []
        unique_chain_ids = torch.unique(chain_ids)
        for chain_id in unique_chain_ids:
            mask = chain_ids == chain_id
            outputs.append(pooler(protein_embed[None, mask, ...]))
        return torch.cat(outputs, dim=0)

    def compute_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor | None:
        """Compute the loss.

        Args:
            logits (torch.FloatTensor): The logits.
            labels (torch.FloatTensor | None, optional): The labels. Defaults
            to None.

        Returns:
            torch.FloatTensor | None: The loss or None if no labels
            are provided.
        """
        if labels is None:
            return None
        return nn.functional.mse_loss(logits, labels)

    def forward(
        self,
        ligand_input_ids: torch.LongTensor,
        receptor_input_ids: torch.LongTensor,
        ligand_attention_mask: torch.LongTensor | None = None,
        receptor_attention_mask: torch.LongTensor | None = None,
        ligand_chain_ids: torch.LongTensor | None = None,
        receptor_chain_ids: torch.LongTensor | None = None,
        labels: torch.FloatTensor | None = None,
    ) -> Dict[str, torch.FloatTensor]:
        """Forward pass of the model.

        Args:
            ligand_input_ids (torch.LongTensor): The ligand input ids.
            receptor_input_ids (torch.LongTensor): The receptor input ids.
            ligand_attention_mask (torch.LongTensor | None, optional): The
            ligand attention mask. Defaults to None.
            receptor_attention_mask (torch.LongTensor | None, optional): The
            receptor attention mask. Defaults to None.
            ligand_chain_ids (torch.LongTensor | None, optional): The ligand
            chain ids. Defaults to None.
            receptor_chain_ids (torch.LongTensor | None, optional): The
            receptor chain ids. Defaults to None.
            labels (torch.FloatTensor | None, optional): The labels.
            Defaults to None.

        Returns:
            Dict[str, torch.FloatTensor]: A dictionary containing the
            logits and the loss.
        """
        # Extract the embeddings
        ligand_embed, receptor_embed = self.backbone(
            ligand_input_ids,
            receptor_input_ids,
            ligand_attention_mask,
            receptor_attention_mask,
        )

        ligand_pooler = (
            self.global_pooler
            if self.shared_global_pooler
            else self.ligand_global_pooler
        )
        receptor_pooler = (
            self.global_pooler
            if self.shared_global_pooler
            else self.receptor_global_pooler
        )

        # Pool the embeddings
        ligand_pooled = ligand_pooler(
            ligand_embed,
            ligand_attention_mask,
        )

        receptor_pooled = receptor_pooler(
            receptor_embed,
            receptor_attention_mask,
        )

        # Process the chains
        if ligand_chain_ids is not None:
            ligand_chains_pooler = (
                self.chains_pooler
                if self.shared_chains_pooler
                else self.ligand_chains_pooler
            )
            ligand_pooled_chains = self.process_chains(
                protein_embed=ligand_pooled,
                chain_ids=ligand_chain_ids,
                pooler=ligand_chains_pooler,
            )
        else:
            ligand_pooled_chains = ligand_pooled

        if receptor_chain_ids is not None:
            receptor_chains_pooler = (
                self.chains_pooler
                if self.shared_chains_pooler
                else self.receptor_chains_pooler
            )
            receptor_pooled_chains = self.process_chains(
                protein_embed=receptor_pooled,
                chain_ids=receptor_chain_ids,
                pooler=receptor_chains_pooler,
            )
        else:
            receptor_pooled_chains = receptor_pooled

        # Aggregate the chains
        aggregated_embed = aggregate_chains(
            sequence_1=ligand_pooled_chains,
            sequence_2=receptor_pooled_chains,
            aggregation_method=self.aggregation_method,
        )

        # Apply the FFN if it is enabled
        if self.use_ffn:
            aggregated_embed = self.ffn(aggregated_embed)

        # Compute the logits and the loss
        logits = self.output(aggregated_embed)
        loss = self.compute_loss(logits, labels)

        # Return the logits and the loss
        return {
            "logits": logits,
            "loss": loss,
        }
