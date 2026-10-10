import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Attention import *
def calc_cosine_similarity(feat1, feat2):
    """
    feat1: [batch, m, dim]
    feat2: [batch, n, dim]
    """
    batch, m, dim = feat1.size()
    n = feat2.size(1)
    fea1_reshape = feat1.reshape(-1, dim)
    fea2_reshape = feat2.reshape(-1, dim)
    fea1_norm = fea1_reshape / torch.norm(fea1_reshape, dim=1, keepdim=True)
    fea2_norm = fea2_reshape / torch.norm(fea2_reshape, dim=1, keepdim=True)
    cosine_similarity = torch.mm(fea1_norm, fea2_norm.t())
    # cosine_similarity = cosine_similarity.reshape(batch, m, n)
    return cosine_similarity

class PromptFormer(nn.Module):
    def __init__(self, sqlen, n_proto, D_e, prompt_dim, args):
        super(PromptFormer, self).__init__()
        self.sqlen = sqlen
        self.n_proto = n_proto
        self.D_e = D_e
        self.prompt_dim = prompt_dim

        self.proj_n = Mlp(in_features=sqlen, hidden_features=n_proto, out_features=n_proto, drop=args.drop_rate)
        self.proj_d = Mlp(in_features=D_e, hidden_features=prompt_dim, out_features=prompt_dim, drop=args.drop_rate)

        self.proj_prompt = nn.Linear(D_e, prompt_dim)

    def _prefix_prototypes(self, x):
        """Evaluate proj_n on every prefix while sharing its original parameters."""
        # x: [batch, seqlen, dim], fc1.weight: [n_proto, seqlen]
        contributions = (
            x.transpose(1, 2).unsqueeze(2)
            * self.proj_n.fc1.weight.unsqueeze(0).unsqueeze(0)
        )  # [batch, dim, n_proto, seqlen]
        hidden = contributions.cumsum(dim=-1).permute(0, 3, 1, 2)
        if self.proj_n.fc1.bias is not None:
            hidden = hidden + self.proj_n.fc1.bias
        hidden = self.proj_n.act(hidden)
        hidden = self.proj_n.drop(hidden)
        prototype = self.proj_n.fc2(hidden)
        prototype = self.proj_n.drop(prototype)
        return prototype.permute(0, 1, 3, 2)  # [batch, seqlen, n_proto, dim]

    def _prompt_from_prefix_prototypes(self, x, umask, prototype):
        batch_size, seqlen, _ = x.shape
        shared_prototype = prototype.permute(1, 0, 2, 3).reshape(
            seqlen, batch_size * self.n_proto, self.D_e
        )
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        prototype_norm = shared_prototype / torch.norm(
            shared_prototype, dim=-1, keepdim=True
        )
        sim = torch.einsum("bsd,snd->bsn", x_norm, prototype_norm)

        # Preserve the upstream query-mask flattening and additive behavior.
        query_mask = umask.reshape(-1).reshape(batch_size, seqlen)
        sim = sim + (1 - query_mask).unsqueeze(-1) * -10000.0
        sim = F.softmax(sim, dim=-1)
        return torch.einsum("bsn,snd->bsd", sim, shared_prototype)
    
    def forward(self, x, umask):
        """
        x -> [batch, seqlen, dim]
        umask -> [batch, seqlen]
        """
        prototype = self._prefix_prototypes(x)
        prompt = self._prompt_from_prefix_prototypes(x, umask, prototype)

        prompt = self.proj_d(prompt) # [batch, seqlen, prompt_dim]
        res = self.proj_prompt(x)
        prompt = prompt + res # [batch, seqlen, prompt_dim] 

        return prompt, prototype, res


class PromptFormer_2(nn.Module):
    def __init__(self, sqlen, n_proto, D_e, prompt_dim, args):
        super(PromptFormer_2, self).__init__()
        self.sqlen = sqlen
        self.n_proto = n_proto
        self.D_e = D_e
        self.prompt_dim = prompt_dim

        self.proj_n = Mlp(in_features=sqlen, hidden_features=n_proto, out_features=n_proto, drop=args.drop_rate)
        self.proj_d = Mlp(in_features=D_e, hidden_features=prompt_dim, out_features=prompt_dim, drop=args.drop_rate)
        self.proj_prompt = nn.Linear(D_e, prompt_dim)

    def _prompt_from_prefix_prototypes(self, x, umask, prototype):
        batch_size, seqlen, _ = x.shape
        shared_prototype = prototype.permute(1, 0, 2, 3).reshape(
            seqlen, batch_size * self.n_proto, self.D_e
        )
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        prototype_norm = shared_prototype / torch.norm(
            shared_prototype, dim=-1, keepdim=True
        )
        sim = torch.einsum("bsd,snd->bsn", x_norm, prototype_norm)
        query_mask = umask.reshape(-1).reshape(batch_size, seqlen)
        sim = sim + (1 - query_mask).unsqueeze(-1) * -10000.0
        sim = F.softmax(sim, dim=-1)
        return torch.einsum("bsn,snd->bsd", sim, shared_prototype)
    
    def forward(self, x, umask, prototype):
        """
        x -> [batch, seqlen, dim]
        umask -> [batch, seqlen]
        """
        prompt = self._prompt_from_prefix_prototypes(x, umask, prototype)

        prompt = self.proj_d(prompt) # [batch, seqlen, prompt_dim]
        res = self.proj_prompt(x)
        prompt = prompt + res # [batch, seqlen, prompt_dim] 

        return prompt

class ComP(nn.Module): 

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False, lbd = 0.9):
        super(ComP, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate
        self.seq_len = args.seq_len
        self.prompt_dim = 16 # to adjust
        self.n_proto = 10 # to adjust
        self.lbd = lbd # to adjust

        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

       
        self.shared_enc_a = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.shared_enc_t = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.shared_enc_v = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.private_enc_a = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.private_enc_t = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.private_enc_v = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.dec_a = Mlp(in_features=D_e*2, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.dec_t = Mlp(in_features=D_e*2, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)
        self.dec_v = Mlp(in_features=D_e*2, hidden_features=D_e, out_features=D_e, drop=args.drop_rate)

        self.block1 = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=int(depth/2),
                )
        self.block2 = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=int(depth/2),
                )
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)

        self.router = Mlp(in_features=D_e * 3, hidden_features=int(D_e * mlp_ratio), out_features=3, drop=drop_rate)

        
        self.prop_proj_a_1 = nn.Linear(D_e+2*self.prompt_dim, D_e)
        self.prop_proj_v_1 = nn.Linear(D_e+2*self.prompt_dim, D_e)
        self.prop_proj_t_1 = nn.Linear(D_e+2*self.prompt_dim, D_e)
        self.prop_proj_a_2 = nn.Linear(D_e+2*self.prompt_dim, D_e)
        self.prop_proj_v_2 = nn.Linear(D_e+2*self.prompt_dim, D_e)
        self.prop_proj_t_2 = nn.Linear(D_e+2*self.prompt_dim, D_e)
        self.prop_proj_rev_a_1 = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e+2*self.prompt_dim, drop=args.drop_rate)
        self.prop_proj_rev_v_1 = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e+2*self.prompt_dim, drop=args.drop_rate)
        self.prop_proj_rev_t_1 = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e+2*self.prompt_dim, drop=args.drop_rate)
        self.prop_proj_rev_a_2 = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e+2*self.prompt_dim, drop=args.drop_rate)
        self.prop_proj_rev_v_2 = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e+2*self.prompt_dim, drop=args.drop_rate)
        self.prop_proj_rev_t_2 = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e+2*self.prompt_dim, drop=args.drop_rate)

        
        self.prop_former_a_1 = PromptFormer(sqlen=self.seq_len, n_proto=10, D_e=D_e, prompt_dim=self.prompt_dim, args=args)
        self.prop_former_v_1 = PromptFormer(sqlen=self.seq_len, n_proto=10, D_e=D_e, prompt_dim=self.prompt_dim, args=args)
        self.prop_former_t_1 = PromptFormer(sqlen=self.seq_len, n_proto=10, D_e=D_e, prompt_dim=self.prompt_dim, args=args)
        self.prop_former_a_2 = PromptFormer_2(sqlen=self.seq_len, n_proto=10, D_e=D_e, prompt_dim=self.prompt_dim, args=args)
        self.prop_former_v_2 = PromptFormer_2(sqlen=self.seq_len, n_proto=10, D_e=D_e, prompt_dim=self.prompt_dim, args=args)
        self.prop_former_t_2 = PromptFormer_2(sqlen=self.seq_len, n_proto=10, D_e=D_e, prompt_dim=self.prompt_dim, args=args)
           

    def forward(self, inputfeats, label, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        input_features_mask -> ?*[seqlen, batch, 3]
        """
        # print(inputfeats[:,:,:])
        # print(input_features_mask[:,:,1])
        weight_save = []
        # sequence modeling
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        # filtering
        shared_a, private_a = self.shared_enc_a(proj_a), self.private_enc_a(proj_a)
        shared_t, private_t = self.shared_enc_t(proj_t), self.private_enc_t(proj_t)
        shared_v, private_v = self.shared_enc_v(proj_v), self.private_enc_v(proj_v)
        
        proj_a_rec = self.dec_a(torch.cat([shared_a, private_a], dim=-1))
        proj_t_rec = self.dec_t(torch.cat([shared_t, private_t], dim=-1))
        proj_v_rec = self.dec_v(torch.cat([shared_v, private_v], dim=-1))

        proj_atv = torch.cat([proj_a, proj_t, proj_v], dim=-1)
        proj_atv_rec = torch.cat([proj_a_rec, proj_t_rec, proj_v_rec], dim=-1)
        shared_atv = torch.cat([shared_a, shared_t, shared_v], dim=-1)
        private_atv = torch.cat([private_a, private_t, private_v], dim=-1)

        if first_stage:
            # --> [batch, 3*seqlen, dim]
            x_a = self.block1(shared_a, first_stage, attn_mask, 'a')
            x_t = self.block1(shared_t, first_stage, attn_mask, 't')
            x_v = self.block1(shared_v, first_stage, attn_mask, 'v')
            x_a = self.block2(x_a, first_stage, attn_mask, 'a')
            x_t = self.block2(x_t, first_stage, attn_mask, 't')
            x_v = self.block2(x_v, first_stage, attn_mask, 'v')
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            
            out_a_private = self.block1(private_a, first_stage, attn_mask, 'a')
            out_t_private = self.block1(private_a, first_stage, attn_mask, 't')
            out_v_private = self.block1(private_a, first_stage, attn_mask, 'v')
            out_a_private = self.block2(out_a_private, first_stage, attn_mask, 'a')
            out_t_private = self.block2(out_t_private, first_stage, attn_mask, 't')
            out_v_private = self.block2(out_v_private, first_stage, attn_mask, 'v')
            out_a_private = self.nlp_head_a(out_a_private)
            out_t_private = self.nlp_head_t(out_t_private)
            out_v_private = self.nlp_head_v(out_v_private)
            out_atv_private = torch.cat([out_a_private, out_t_private, out_v_private], dim=-1)
            x = torch.cat([x_a, x_t, x_v], dim=1)
            x[attn_mask == 0] = 0
            x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
            x_joint = torch.cat([x_a, x_t, x_v], dim=-1)

        else:

            prompt_a_0, prototype_a, res_a = self.prop_former_a_1(shared_a, input_mask[:,:,0].transpose(0,1))
            prompt_t_0, prototype_t, res_t = self.prop_former_t_1(shared_t, input_mask[:,:,1].transpose(0,1))
            prompt_v_0, prototype_v, res_v = self.prop_former_v_1(shared_v, input_mask[:,:,2].transpose(0,1))
            shared_a = self.prop_proj_a_1(torch.cat([shared_a, prompt_t_0, prompt_v_0], dim=-1))
            shared_t = self.prop_proj_t_1(torch.cat([shared_t, prompt_a_0, prompt_v_0], dim=-1))
            shared_v = self.prop_proj_v_1(torch.cat([shared_v, prompt_a_0, prompt_t_0], dim=-1))

            x_a = self.block1(shared_a, first_stage, attn_mask, 'a')
            x_t = self.block1(shared_t, first_stage, attn_mask, 't')
            x_v = self.block1(shared_v, first_stage, attn_mask, 'v')
            x_a = self.prop_proj_rev_a_1(x_a)
            x_t = self.prop_proj_rev_t_1(x_t)
            x_v = self.prop_proj_rev_v_1(x_v)
            # shared_a, shared_t, shared_v = x_a[:,:,:self.D_e], x_t[:,:,:self.D_e], x_v[:,:,:self.D_e]
            shared_a, prompt_t_a_0, prompt_v_a_0 = x_a[:, :, :self.D_e], x_a[:, :, self.D_e:self.D_e+self.prompt_dim], x_a[:, :, self.D_e+self.prompt_dim:]
            shared_t, prompt_a_t_0, prompt_v_t_0 = x_t[:, :, :self.D_e], x_t[:, :, self.D_e:self.D_e+self.prompt_dim], x_t[:, :, self.D_e+self.prompt_dim:]
            shared_v, prompt_a_v_0, prompt_t_v_0 = x_v[:, :, :self.D_e], x_v[:, :, self.D_e:self.D_e+self.prompt_dim], x_v[:, :, self.D_e+self.prompt_dim:]

            prompt_a_1 = self.prop_former_a_2(shared_a, input_mask[:,:,0].transpose(0,1), prototype_a)
            prompt_t_1  = self.prop_former_t_2(shared_t, input_mask[:,:,1].transpose(0,1), prototype_t)
            prompt_v_1 = self.prop_former_v_2(shared_v, input_mask[:,:,2].transpose(0,1), prototype_v)
            prompt_t_a_1, prompt_v_a_1 = self.lbd*prompt_t_a_0 + (1-self.lbd)*prompt_t_1, self.lbd*prompt_v_a_0 + (1-self.lbd)*prompt_v_1
            prompt_a_t_1, prompt_v_t_1 = self.lbd*prompt_a_t_0 + (1-self.lbd)*prompt_a_1, self.lbd*prompt_v_t_0 + (1-self.lbd)*prompt_v_1
            prompt_a_v_1, prompt_t_v_1 = self.lbd*prompt_a_v_0 + (1-self.lbd)*prompt_a_1, self.lbd*prompt_t_v_0 + (1-self.lbd)*prompt_t_1

            x_a = self.prop_proj_a_2(torch.cat([shared_a, prompt_t_a_1, prompt_v_a_1], dim=-1))
            x_t = self.prop_proj_t_2(torch.cat([shared_t, prompt_a_t_1, prompt_v_t_1], dim=-1))
            x_v = self.prop_proj_v_2(torch.cat([shared_v, prompt_a_v_1, prompt_t_v_1], dim=-1))
            x_a = self.block2(x_a, first_stage, attn_mask, 'a')
            x_t = self.block2(x_t, first_stage, attn_mask, 't')
            x_v = self.block2(x_v, first_stage, attn_mask, 'v')

            x_out_a = x_a
            x_out_t = x_t
            x_out_v = x_v

            out_a = self.nlp_head_a(x_out_a)
            out_t = self.nlp_head_t(x_out_t)
            out_v = self.nlp_head_v(x_out_v)

            x = torch.cat([x_out_a, x_out_t, x_out_v], dim=-1)# [batch, seq_len, 3 * D_e]
            weights = torch.softmax(self.router(x),dim=-1)# [batch, seq_len, 3]
            weights = weights.unsqueeze(-1)  # [batch, seq_len, 3, 1]
            x_unweighted = torch.stack([x_a, x_t, x_v], dim=2)  # [batch, seq_len, 3, D_e]
            x_out = weights * x_unweighted  # [batch, seq_len, 3, D_e]
            x_out_a, x_out_t, x_out_v = torch.unbind(x_out, dim=2)
            x_joint = torch.cat([x_out_a, x_out_t, x_out_v], dim=-1)

            out_a_private, out_t_private, out_v_private = 0, 0, 0 #仅做占位用
            out_atv_private = 0


        
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        out = self.nlp_head(hidden)

        

        return hidden, out, out_a, out_t, out_v, np.array(weight_save), proj_atv, proj_atv_rec, shared_atv, private_atv, out_atv_private


if __name__ == '__main__':
    input = [torch.randn(61, 32, 300)]
    model = ComP(100, 100, 100, 128, 1)
    anchor = torch.randn(32, 61, 128)
    hidden, out, _ = model(input)
