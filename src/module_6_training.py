"""
module_6_training.py
-----------------------------------------
Orchestrates the training loop for the entire pipeline:
  - token embedding
  - positional encoding
  - stock+sector identity embedding
  - sequence model (transformer)
  - neural SDE
  - final output head (maps SDE hidden state -> label_dim)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
from torch.utils.data import DataLoader

from module_7_embeddings import TokenEmbedding
from module_8_positional_encoding import PositionalEncoding
from module_9_identity_embedding import IdentityEmbedding
from module_10_sequence_model import SequenceModel
from module_11_neural_sde import NeuralSDE, SDEFunction
from module_12_output_head import FinalHead  # <- newly added

class SDETrainer:
    """
    Orchestrates the entire pipeline.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Build modules
        self.token_embed = TokenEmbedding(args.token_dim, args.embed_dim).to(self.device)
        self.pos_encoder = PositionalEncoding(d_model=args.embed_dim, max_len=args.sequence_length).to(self.device)
        self.identity_embed = IdentityEmbedding(args.num_stocks, args.num_sectors, args.embed_dim).to(self.device)
        self.seq_model = SequenceModel(args.embed_dim, args.num_heads, args.num_layers, args.dropout).to(self.device)
        self.neural_sde = NeuralSDE(args.embed_dim, args.param_dim_f, args.param_dim_g).to(self.device)
        
        # 2) The final head mapping from embed_dim -> label_dim
        # e.g. if you want 5 output tasks, set label_dim=5
        self.final_head = FinalHead(args.embed_dim, args.label_dim).to(self.device)

        # 3) Combine parameters
        all_params = (
            list(self.token_embed.parameters())
            + list(self.pos_encoder.parameters())
            + list(self.identity_embed.parameters())
            + list(self.seq_model.parameters())
            + list(self.neural_sde.parameters())
            + list(self.final_head.parameters())  # Add final_head
        )
        self.optimizer = optim.Adam(all_params, lr=args.lr)

        # By default, MSE to handle multi-dim label: shape [B, label_dim]
        self.criterion = nn.MSELoss()

    def train_one_epoch(self, train_loader: DataLoader):
        # Set modules in training mode
        self.token_embed.train()
        self.pos_encoder.train()
        self.identity_embed.train()
        self.seq_model.train()
        self.neural_sde.train()
        self.final_head.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # batch: (tokens, stock_ids, sector_ids, labels)
            # - tokens: [B, T, token_dim]
            # - stock_ids: [B]
            # - sector_ids: [B]
            # - labels: [B, label_dim] (e.g., 5)
            tokens, stock_ids, sector_ids, labels = batch
            tokens = tokens.to(self.device)
            stock_ids = stock_ids.to(self.device)
            sector_ids = sector_ids.to(self.device)
            labels = labels.to(self.device)  # shape [B, label_dim]

            self.optimizer.zero_grad()

            # 1) Embeddings
            h = self.token_embed(tokens)          # [B, T, embed_dim]
            h = self.pos_encoder(h)               # [B, T, embed_dim]

            # 2) Identity embedding
            id_emb = self.identity_embed(stock_ids, sector_ids)   # [B, embed_dim]
            id_emb = id_emb.unsqueeze(1).expand(-1, h.size(1), -1)  # [B, T, embed_dim]
            z = h + id_emb

            # 3) Transformer-based sequence model
            seq_out = self.seq_model(z)           # [B, T, embed_dim]

            # 4) SDE integration
            H0 = seq_out[:, 0, :]  # shape [B, embed_dim]
            sde_times = torch.linspace(0, 1, seq_out.size(1), device=self.device)
            sde_func = SDEFunction(self.neural_sde)
            # H_traj: [seq_len, B, embed_dim]
            H_traj = torchsde.sdeint(sde_func, H0, sde_times, method='euler')
            H_final = H_traj[-1]  # [B, embed_dim]

            # 5) Predict label from hidden state
            # e.g. shape => [B, label_dim]
            predicted_label = self.final_head(H_final)

            # 6) Loss
            loss = self.criterion(predicted_label, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_size = tokens.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(1, total_samples)

    def evaluate(self, val_loader: DataLoader):
        # Set modules in eval mode
        self.token_embed.eval()
        self.pos_encoder.eval()
        self.identity_embed.eval()
        self.seq_model.eval()
        self.neural_sde.eval()
        self.final_head.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                tokens, stock_ids, sector_ids, labels = batch
                tokens = tokens.to(self.device)
                stock_ids = stock_ids.to(self.device)
                sector_ids = sector_ids.to(self.device)
                labels = labels.to(self.device)  # shape [B, label_dim]

                # Forward pass
                h = self.token_embed(tokens)
                h = self.pos_encoder(h)
                id_emb = self.identity_embed(stock_ids, sector_ids)
                id_emb = id_emb.unsqueeze(1).expand(-1, h.size(1), -1)
                z = h + id_emb

                seq_out = self.seq_model(z)
                H0 = seq_out[:, 0, :]
                sde_times = torch.linspace(0, 1, seq_out.size(1), device=self.device)
                sde_func = SDEFunction(self.neural_sde)
                H_traj = torchsde.sdeint(sde_func, H0, sde_times, method='euler')
                H_final = H_traj[-1]

                predicted_label = self.final_head(H_final)
                loss = self.criterion(predicted_label, labels)

                batch_size = tokens.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / max(1, total_samples)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        best_val_loss = float('inf')
        no_improve = 0

        for epoch in range(self.args.num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            print(f"Epoch {epoch+1}/{self.args.num_epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                no_improve += 1
                if no_improve >= self.args.patience:
                    print("Early stopping triggered.")
                    break

    def parameters(self):
        # Combine all parameters for gradient steps
        return list(self.token_embed.parameters()) \
               + list(self.pos_encoder.parameters()) \
               + list(self.identity_embed.parameters()) \
               + list(self.seq_model.parameters()) \
               + list(self.neural_sde.parameters()) \
               + list(self.final_head.parameters())

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.args.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}, val_loss={val_loss:.4f}")

    def state_dict(self):
        return {
            'token_embed': self.token_embed.state_dict(),
            'pos_encoder': self.pos_encoder.state_dict(),
            'identity_embed': self.identity_embed.state_dict(),
            'seq_model': self.seq_model.state_dict(),
            'neural_sde': self.neural_sde.state_dict(),
            'final_head': self.final_head.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
