import random
import torch
import torch.nn as nn

from model.revgrad import GradReversal
from utils.model_utils import load_gloveembeddings, project_vector
from model.attention import Attention, GAT
from model.encoder import Encoder
from model.decoder import Decoder
from model.classifier import DomainClassifier

from configs.config import HP

import re

class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()

        self.hp = HP()  # Hyperparametters
        self.device = self.hp.device
        self.vocab = vocab
        self.src_pad_idx = vocab.pad()
        self.input_dim = len(vocab)
        self.output_dim = len(vocab)
        self.num_tgr_docs_domains = self.hp.num_tgr_docs_domains
        self.concat_docs = self.hp.concat_docs

        ##############################################
        # Embeddings
        ##############################################
        if self.hp.use_pretrained_embs:
            emb_vecs = load_gloveembeddings(vocab, Glove_name=self.hp.glove_name, Glove_dim=self.hp.emb_dim)
            self.embedding = nn.Embedding.from_pretrained(emb_vecs, freeze=False, padding_idx=self.src_pad_idx)
        else:
            self.embedding = nn.Embedding(self.input_dim, self.hp.emb_dim, padding_idx=self.src_pad_idx)

        ##############################################
        # Attention Layers
        ##############################################

        # Decoder Attention layer
        self.dec_attn = Attention(enc_dim=self.hp.enc_hidden, dec_dim=self.hp.dec_hidden)

        # Graph Attention Layer (GAT)
        if self.hp.use_gat:
            self.gat_rec = GAT(device=self.device,
                               input_dim=self.hp.enc_hidden * self.hp.enc_layers,
                               output_dim=self.hp.enc_hidden * self.hp.enc_layers,
                               hidden_dim=self.hp.gat_hidden,
                               dropout=self.hp.gat_drop,
                               alpha=self.hp.gat_alpha,
                               num_heads=self.hp.gat_heads)
            if self.hp.use_cls:
                self.gat_cls = GAT(device=self.device,
                                   input_dim=self.hp.enc_hidden * self.hp.enc_layers,
                                   output_dim=self.hp.enc_hidden * self.hp.enc_layers,
                                   hidden_dim=self.hp.gat_hidden,
                                   dropout=self.hp.gat_drop,
                                   alpha=self.hp.gat_alpha,
                                   num_heads=self.hp.gat_heads)
            else:
                self.gat_cls = None

        ##############################################
        # Encoders
        ##############################################
        self.encoder_rec = Encoder(device=self.device,
                                   input_dim=self.input_dim,
                                   emb_dim=self.hp.emb_dim,
                                   enc_dim=self.hp.enc_hidden,
                                   dec_dim=self.hp.dec_hidden,
                                   num_layers=self.hp.enc_layers,
                                   gat_attn=self.gat_rec,
                                   dropout=self.hp.enc_drop)
        if self.hp.use_cls:
            self.encoder_rec = Encoder(device=self.device,
                                       input_dim=self.input_dim,
                                       emb_dim=self.hp.emb_dim,
                                       enc_dim=self.hp.enc_hidden,
                                       dec_dim=self.hp.dec_hidden,
                                       num_layers=self.hp.enc_layers,
                                       gat_attn=self.gat_cls,
                                       dropout=self.hp.enc_drop)

        ##############################################
        # Decoders
        ##############################################
        self.decoder = Decoder(device=self.device,
                               output_dim=self.output_dim,
                               emb_dim=self.hp.emb_dim,
                               enc_dim=self.hp.enc_hidden,
                               dec_dim=self.hp.dec_hidden,
                               attention=self.dec_attn,
                               dropout=self.hp.dec_drop)

        ##############################################
        # Classifier
        ##############################################
        # used to reduce the dimension of the classifier input
        if self.num_target_docs > 1:
            self.fc = nn.Linear(self.hp.dec_hidden * self.num_target_docs, self.hp.dec_hidden)
        self.classifier = DomainClassifier(input_dim=self.hp.dec_hidden, hidden_dim=self.hp.cls_hidden)

    def filter_oov(self, tensor):
        """ Replace any OOV index in `tensor` with <unk> token """
        result = tensor.clone()
        result[tensor >= len(self.vocab)] = self.vocab.unk()
        return result

    def encode(self, src_senti, src_embedded, src_len, src_max_len, gamma=0.5):
        batch_size = src_embedded.size(1)
        # We encode the documents using the rec_encoder to train the language model
        rec_enc_outputs, H = self.encoder_rec(src_embedded, src_len, src_max_len)
        # rec_enc_outputs: [seq_len, batch_size, enc_dim*2]
        # H: text feature representations (full information) for language model [batch_size, dec_dim]

        if self.hp.use_cls:
            # For the domain (sentiment) classification branch, we pair each review with one or a concatenation
            # of reviews having a different (opposite) sentiment than the target review. We then cacatenate both
            # representations to be fed to the domain classifier
            # H_hat_inv: domain-shared text feature representations after GRL

            # First, encode all reviews in the batch
            cls_enc_outputs, H_hat_inv = self.encoder_cls(src_embedded, src_len, src_max_len)  # [batch_size, dec_dim]
            # H_hat_inv: text feature representations (full information) for classifier
            
            if self.num_tgr_docs_domains == 1:
                hiddens = sample_target_docs(H_hat_inv, batch_size, domain="opposite", concat=self.concat_docs) # [batch_size, dec_dim]
            elif self.num_tgr_docs_domains == 2:
                same_hiddens = sample_target_docs(in_hidden, batch_size, domain="same", concat=self.concat_docs)
                opp_hiddens = sample_target_docs(in_hidden, batch_size, domain="opposite", concat=self.concat_docs)
                hiddens = torch.concat([opp_hiddens, same_hiddens], dim=1) # [batch_size, dec_dim*2]
                hiddens = self.fc(hiddens) # [batch_size, dec_dim]
                
            H_hat_inv = hiddens
            
            # Gradient Reversal Layer
            if self.hp.use_grl:
                 H_hat_inv = GradReversal.grad_reverse(H_hat_inv, alpha=gamma)  # [batch_size, dec_dim]
                    
            # Sentiment classification
            senti_preds = self.classifier(H_hat_inv).squeeze(1)  # [batch_size]
            
            # Projection Mechanism
            if self.hp.use_proj:
                H_hat = project_vector(H, H_hat_inv, self.device)  # [batch_size, dec_dim]
                H_tilt = project_vector(H, H - H_hat, self.device)  # [batch_size, dec_dim]
                # H_hat: domain-shared text feature representations after GRL
                # H_tilt: domain-specific text feature representations
                
        if self.hp.dec_hidden_type == 0:
            enc_hiddens = H
            enc_outputs = rec_enc_outputs
        elif self.hp.dec_hidden_type == 1:
            enc_hiddens = H_hat
            enc_outputs = cls_enc_outputs
        else:
            enc_hiddens = H_tilt
            enc_outputs = rec_enc_outputs
               
        return senti_preds, enc_hiddens, enc_outputs

    def decode(self, src_input, src_len, enc_hidden, teacher_forcing_ratio=0.5, train=True):
        # src_input: [seq_len, batch_size]
        batch_size = src_input.size(1)

        # first input to the decoder is the <sos> tokens
        dec_input = src_input[0, :]  # [batch_size]
        gen_len = src_len.max()

        final_dists = [torch.zeros(batch_size, self.output_dim).to(self.device)]

        for t in range(1, gen_len):
            dec_input = self.embedding(self.filter_oov(dec_input.unsqueeze(0)))
            vocab_dist, enc_hidden = self.decoder(dec_input, enc_hidden)
            # vocab_dist: [batch_size, vocab_size]
            # attn_dists: [batch_size, seq_len]
            # enc_hidden: [batch_size, dec_dim]

            # get the highest predicted token from our predictions
            top1 = vocab_dist.argmax(1)
            
            # decide if we are going to use teacher forcing or not
            if not train or teacher_forcing_ratio is None:
                teacher_force = False
            else:
                teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input. If not, use predicted token
            dec_input = src_input[t] if teacher_force else top1

            # append to a list all interesting results for end output and error calculation
            final_dists.append(vocab_dist)

        final_dists = torch.stack(final_dists, dim=-1)  # [batch_size, ext_vocab, gen_len]

        return final_dists

    def forward(self, src_input, src_senti, src_len, gamma=0.5, tf_ratio=0.5):
        # src_input: [seq_len, batch_size]
        # src_senti: [batch_size]
        # src_len: [1, batch_size]
        #src_input = src_input.permute(1, 0)  # [seq_len, batch_size]
        src_max_len = src_len.max().clone().detach().repeat(src_len.size())  # [batch_size]

        #### Embeddings ####
        src_embedded = self.embedding(src_input)  # [seq_len, batch_size, emb_dim]

        #### Encoding ####
        senti_preds, enc_hiddens, enc_outputs = self.encode(src_senti, src_embedded, src_len, src_max_len, gamma)

        #### Decoder ####
        rec_final_dists = self.decode(src_input, src_len, enc_hiddens, tf_ratio)

        return senti_preds, rec_final_dists
    
    
def sample_target_docs(in_hidden, batch_size, domain="same", concat=False):
    for idx in range(batch_size):
        if domain == "same":
            h = hidden[src_domain != src_domain[idx]] # [num target docs in batch, dec_dim]
        elif domain == "opposite":
            h = hidden[src_domain == src_domain[idx]] # [num source docs in batch, dec_dim]
            h = h[[i != idx for i, _ in enumerate(h)]]  # [num source docs in batch - 1, dec_dim]
        else:
            raise Error(f"domain: '{domain}' is not recognized. expected: all, same, or opposite.")
        
        if concat:
            h = torch.mean(h, dim=0).unsqueeze(0)
        else:
            h = h[random.randint(0, h.size(0) - 1)].unsqueeze(0)
        
        out_hidden.append(h)
    
    out_hidden = torch.vstack(out_hidden)
    return out_hidden