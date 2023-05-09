import torch
import torch.nn as nn

from models import DiscreteVAE, return_mask_lengths


class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.vae = DiscreteVAE(args).to(self.device)
        params = filter(lambda p: p.requires_grad, self.vae.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_d_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_zd_kl = 0
        self.loss_info = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions,d1_ids,d2_ids,d3_ids):
        d_id_list = [d1_ids,d2_ids,d3_ids]
        chosen_did = torch.randint(low=0,high=3,size=(1,))[0].item()
        #print("\n\nchosen did: ", chosen_did)
        d_ids = d_id_list[chosen_did]
        self.vae = self.vae.train()
        # Forward
        loss, \
        loss_q_rec, loss_a_rec, \
        loss_zq_kl, loss_za_kl, \
        loss_info, loss_d_rec, loss_zd_kl \
        = self.vae(c_ids, q_ids, a_ids, start_positions, end_positions,d_ids)
        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Step
        self.optimizer.step()

        self.loss_q_rec = loss_q_rec.item()
        self.loss_a_rec = loss_a_rec.item()
        self.loss_zq_kl = loss_zq_kl.item()
        self.loss_za_kl = loss_za_kl.item()
        self.loss_zd_kl = loss_zd_kl.item()
        self.loss_info = loss_info.item()
        self.loss_d_rec = loss_d_rec.item()

    def generate_posterior(self, c_ids, q_ids, a_ids,d_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za, _, _, zd = self.vae.posterior_encoder(c_ids, q_ids, a_ids,d_ids)
            q_ids, start_positions, end_positions, d_ids = self.vae.generate(zq, za, c_ids, zd)
        return q_ids, start_positions, end_positions, zq, d_ids, zd

    def generate_answer_logits(self, c_ids, q_ids, a_ids, d_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za, _, _, zd = self.vae.posterior_encoder(c_ids, q_ids, a_ids, d_ids)
            start_logits, end_logits = self.vae.return_answer_logits(zq, za, c_ids, zd)
        return start_logits, end_logits

    def generate_prior(self, c_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za, _, _, zd = self.vae.prior_encoder(c_ids)
            q_ids, start_positions, end_positions, d_ids = self.vae.generate(zq, za, c_ids, zd)
        return q_ids, start_positions, end_positions, zq, d_ids, zd    

    def save(self, filename):
        params = {
            'state_dict': self.vae.state_dict(),
            'args': self.args
        }
        torch.save(params, filename)
