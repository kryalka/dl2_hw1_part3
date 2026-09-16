import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


class TromptCell(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model):
        super().__init__()
        # Embeddings (Figure 3.2)
        self.feature_emb_weight = nn.Parameter(torch.empty(n_columns, d_model))
        self.feature_emb_bias = nn.Parameter(torch.empty(n_columns, d_model))
        self.ln_emb = nn.LayerNorm(d_model)

        # Importance Getter (Figure 3.1)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model)

        self.emb_column = nn.Parameter(torch.empty(n_columns, d_model))
        self.emb_prompt = nn.Parameter(torch.empty(n_prompts, d_model))

        # Modified expansion block (Figure 3.3), without non-linearities
        self.dense_expand = nn.Linear(1, n_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.feature_emb_weight.shape[1] ** -0.5
        nn.init.uniform_(self.feature_emb_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.feature_emb_bias, -d_rsqrt, d_rsqrt)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)

    def forward(self, x: torch.Tensor, prev_cell_out: torch.Tensor) -> torch.Tensor:
        with record_function("cell_feature_embedding"):
            x_emb = x.unsqueeze(-1) * self.feature_emb_weight + self.feature_emb_bias.unsqueeze(0)
            x_emb = F.relu(x_emb)
            x_emb = self.ln_emb(x_emb)

        with record_function("cell_importance_getter"):
            with record_function("prompt_normalization"):
                x_prompt = self.emb_prompt

            with record_function("importance_dense"):
                x_prompt = self.dense_imp(torch.cat([self.ln_prompt(x_prompt), prev_cell_out], dim=-1)) + x_prompt

            with record_function("column_normalization"):
                x_column = self.ln_col(self.emb_column)

            with record_function("attention_scores"):
                scores = x_prompt @ x_column.T

            with record_function("attention_softmax"):
                mask = torch.softmax(scores, dim=-1)

        with record_function("cell_weighted_sum"):
            a = self.dense_expand.weight[:, 0].view(1, -1, 1)
            q = self.dense_expand.bias.view(1, -1, 1)
            x_out = (1.0 + a) * (mask @ x_emb) + q

        return x_out


class TromptDownstream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pw = torch.softmax(self.dense0(x).squeeze(-1), dim=-1)
        xnew = (pw.unsqueeze(-1) * x).sum(dim=-2)
        return self.dense_out(self.ln(F.relu(self.dense1(xnew))))


class Trompt(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model, n_cycles):
        super().__init__()
        self.tcells = nn.ModuleList([TromptCell(n_columns, n_prompts, d_model) for _ in range(n_cycles)])
        self.tdown = TromptDownstream(d_model)
        self.prompt = nn.Parameter(torch.empty(n_prompts, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, x):
        x_prompt = self.prompt
        outputs = []
        for cell in self.tcells:
            outputs.append(self.tdown(cell(x, x_prompt)))
        return torch.stack(outputs, dim=1).squeeze(-1)
