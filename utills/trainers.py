import os
import random
import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm

from utills.step_points import Shuff_Reshuff

class R_trainer():
        def __init__(self,
                      model,
                      device,
                      path_save: str,
                      next_train = False,
                      path_cpt = None,

                      ):
              super(R_trainer, self).__init__()
              self.path_save = path_save
              self.next_train = next_train
              self.DEVICE = device
              self.model = model
              self.emb = model.emb_dim
              self.model_class = self.model.__class__.__name__

              if self.next_train:
                  self.checkpoint = torch.load(path_cpt)
                  self.model.load_state_dict(self.checkpoint['model_state_dict'])
                  self.last_save = self.checkpoint['saved_model']
                  self.last_epoch = self.checkpoint['epoch']
                  self.last_lr = self.checkpoint["last_history"]["lr"][-1] #
                  self.best_loss = self.checkpoint['loss']
                  self.best_loss_mse = self.checkpoint['mse_loss']
                  self.best_loss_cos = self.checkpoint['cos_loss']
                  self.best_acc = self.checkpoint['acc']
                  self.history_train = self.checkpoint['all_history']
                  self.best_eph = {"loss" : self.last_epoch,
                                  "cos" : self.last_epoch,
                                  "mse" : self.last_epoch,
                                  "acc" : self.last_epoch,
                                  }
                  self.last_checkpoint =  str(path_cpt.split("/")[-1])
              # state for start
              else:
                  self.last_checkpoint = "New"
                  self.last_save = ''
                  self.last_epoch = 0
                  self.best_loss = 1000000
                  self.best_loss_mse = 1000000
                  self.best_loss_cos = 1000000
                  self.best_acc = 0
                  self.best_eph = {"loss" : 0,
                                  "cos" : 0,
                                  "mse" : 0,
                                  "acc" : 0,
                                  }

                  self.history_train = {"loss" : [],
                                        "loss_cos" : [],
                                        "loss_mse" : [],
                                        "acc" : [],
                                        "lr" : [],
                                        "base_loss" : []
                                        }

                  self.best_eph = {"loss" : 0,
                                    "cos" : 0,
                                    "mse" : 0,
                                    "acc" : 0
                                    }

              self.hist = {"loss" : [],
                          "loss_cos" : [],
                          "loss_mse" : [],
                          "acc" : [],
                          "base_loss" : [],
                          "lr" : []
                  }

              self.logs = {"bad_movi" : [],
                          "empty" : [],
                          "naninf_loss" : [],
                          "naninf_mse" : [],
                          "naninf_acc" : [],
                          "naninf_cos" : []
                  }

              # paths to save temory weights
              PATH_model_loss = self.path_save +'/last_best_loss.pt'
              PATH_model_cos = self.path_save +'/last_best_cos.pt'
              PATH_model_mse = self.path_save +'/last_best_mse.pt'
              PATH_model_acc = self.path_save +'/last_best_acc.pt'

              foder_temp_checkpoints = self.path_save + "/temp_checkpoints"
              if not os.path.exists(foder_temp_checkpoints):
                   os.makedirs(foder_temp_checkpoints)

              self.dict_model_paths = {"loss" : PATH_model_loss,
                                        "cos" : PATH_model_cos,
                                        "mse" : PATH_model_mse,
                                        "acc" : PATH_model_acc,
                                        "temp": foder_temp_checkpoints
                                        }

              # init waitings jdun
              self.wait_train = 0
              self.wait2end = 0

        def trailoop(self,
                     train_data,
                     max_batch : int,
                     optimizer: object,
                     scheduler: object,
                     maker_points: object,
                     rotator: object,
                     epochs: int,
                     friq_save_checkpoint = 15,
                     jdun_train = 3,
                     jdun_end = 3,
                     use_on_epoch = .9,
                     exp_rate = 0.99,
                     window = 3,
                     update_best = 0,
                     learning_rate = 1e-03,
                     update_lr = False,
                     reset_losses_acc = False,
                     suff_direct =  True,
                     add_back_train = False,
                     add_rote_train = False,
                     add_diff_train = False,
                     change_name_to_save = None,
                     ):

            # correct save, train and lerning rate
            self.suff_direct =  suff_direct
            self.add_back_train = add_back_train
            self.add_rote_train = add_rote_train
            self.add_diff_train = add_diff_train
            self.fr_sv_cpt = friq_save_checkpoint
            self.window = window
            self.JDUN_END = jdun_end
            self.JDUN_TRAIN = jdun_train
            self.max_batch = max_batch
            self.use_on_epoch = use_on_epoch
            self.train_data = train_data
            self.EPOCHS = epochs
            self.update_lr = update_lr
            self.LR_RATE = learning_rate
            self.EXP_RATE = exp_rate
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.GAMMA = self.scheduler.gamma
            self.reset_losses_acc = reset_losses_acc
            self.change_name_to_save = change_name_to_save

            # https://blog.paperspace.com/pytorch-loss-functions/
            self.cos_loss = nn.CosineEmbeddingLoss(reduction='none').to(self.DEVICE)
            self.mse_loss = nn.MSELoss(reduction='none').to(self.DEVICE)
            self.target = (-1) * torch.ones(self.emb).to(self.DEVICE)

            # ComputeDiffPoints class
            self.maker_points = maker_points

            # inite rotation class
            self.RV = rotator()
            
            # wait to update_best
            self.update_best = update_best + self.window


            # correction for next_train
            if self.next_train:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.LR_RATE  if self.update_lr else self.last_lr
                self.scheduler.last_epoch = self.last_epoch

                if self.reset_losses_acc:
                      self.last_save = ''
                      self.last_epoch = 0
                      self.best_loss = 1000000
                      self.best_loss_mse = 1000000
                      self.best_loss_cos = 1000000
                      self.best_acc = 0
                      self.best_eph = {"loss" : 0,
                                      "cos" : 0,
                                      "mse" : 0,
                                      "acc" : 0,
                                      }



            for epoch  in tqdm(range(self.last_epoch, self.last_epoch + self.EPOCHS),
                               unit ="EPOHS", desc ="Пробегаемся по всем эпохам"):
                # inite
                self.flush_memory()
                text = ''
                save_model = 0
                eph_loss = 0
                eph_loss_cos = 0
                eph_loss_mse = 0
                eph_cos_acc = 0
                eph_base_loss = 0
                cos_acc = 0
                cur_lr = self.optimizer.param_groups[0]['lr']
                diff_loss_weight = 0

                random.shuffle(self.train_data)  # shuffle  data each epoch
                take = int(len(self.train_data)*self.use_on_epoch)
                take_data = self.train_data[:take]  # take shuffle part data each epoch

                # go by selected data
                for  id_m, data  in tqdm(enumerate(take_data), unit = f" movi ",
                                                    desc ="Пробегаемся по всем фильмам"):
                    self.flush_memory()
                    # get next frame embbedings
                    id_movi, ids_frame = data['id_movi'], data['ids_frames']
                    text_hid_state, unclip_embed  = data['last_hidden_state'], data['text_embed']
                    image_embeds = data['img_embeds']

                    # control long movi
                    qty_frames = len(image_embeds)
                    d_batch = self.max_batch-1
                    if qty_frames and id_movi:
                        if qty_frames <= self.max_batch:
                            d_batch = qty_frames-1

                        # take random ids in ids_frame[1:] because ids_frame[0] will first
                        rand_ids = [0] + list(np.random.choice(np.arange(1, qty_frames-1),
                                                                d_batch-1, replace=False))
                        rand_ids = rand_ids + [qty_frames-1]
                        # select rand_ids labels and image_embeds
                        image_embeds = torch.concat([image_embeds[i] for i in rand_ids]).unsqueeze(dim=1)
                        labels = np.array([ids_frame[i] for i in rand_ids])

                        # place labels to class points
                        self.maker_points.time_labels = labels
                        config_norm, config_back = self.maker_points.getpoints_train()
                        all_points = self.maker_points.points

                        if self.add_back_train:
                            all_points = np.append(all_points, self.maker_points.back_points)

                        # intit class for shufflee
                        b_srs = Shuff_Reshuff(len(all_points))
                        if self.suff_direct:
                            bs_idx = b_srs.shuffle() # shuffleed indexees
                            bu_idx = b_srs.unshuffle() # indexees for re_shuffle
                        else:
                            bs_idx = b_srs.idx_base # shuffleed indexees
                            bu_idx = b_srs.idx_base # indexees for re_shuffle

                        ######### compute enters and states for model and losses for base ways

                        # collect time directions tensors
                        time_base = torch.tensor(all_points).unsqueeze(1)
                        directions = [torch.zeros_like(time_base[:d_batch])]
                        if self.add_back_train:
                            directions.append(torch.ones_like(time_base[d_batch:]))
                        directions = torch.concat(directions)
                        time_base = time_base.to(self.DEVICE)
                        directions = directions.to(self.DEVICE)

                        # collect text tensors
                        text_hid_states = [text_hid_state for _ in range(d_batch)]
                        if self.add_back_train:
                            text_hid_states.extend([text_hid_state for _ in range(d_batch)])
                        text_hid_states =  torch.concat(text_hid_states).to(self.DEVICE)

                        # collect unclip tensors
                        base_unclip_embs = [unclip_embed for _ in range(d_batch)]
                        if self.add_back_train:
                            base_unclip_embs.extend([unclip_embed for _ in range(d_batch)])
                        base_unclip_embs = torch.concat(base_unclip_embs).unsqueeze(dim=1).to(self.DEVICE)

                        # collect base_img_embs tensors
                        base_img_embs = [image_embeds[0] for _ in range(d_batch)]
                        if self.add_back_train:
                            base_img_embs.extend([image_embeds[-1] for _ in range(d_batch)])
                        base_img_embs = torch.concat(base_img_embs).unsqueeze(dim=1)

                        # collect img_embs tensors
                        img_embs = image_embeds[1:]

                        if self.add_back_train:
                            # collect back_img_embs tensors
                            back_img_embs = torch.flip(image_embeds, [0,])[1:]
                            # collect img_embs together tensors
                            img_embs = torch.concat([img_embs, back_img_embs])
                        ####################################################################

                        # img difference
                        diff_img_embs =  (base_img_embs[bs_idx].squeeze(dim=1).to(torch.float32) - img_embs[bs_idx].squeeze(dim=1).to(torch.float32)) #
                        # check on static movi
                        bad_movi = 0
                        for k in range(diff_img_embs.shape[0]):
                          if abs(diff_img_embs[k].sum()) ==0:
                            bad_movi+=1

                        if not bad_movi:
                            # zero_grad optimizer
                            self.optimizer.zero_grad()

                            # base predict which can used in next step
                            pred_unclip_embs = self.model(
                              text_hidden_states = text_hid_states[bs_idx].to(torch.float32), # shuffleed
                              prior_embeds = base_unclip_embs[bs_idx].to(torch.float32),
                              delta_time = time_base[bs_idx],
                              direction = directions[bs_idx]
                                                )

                            # difference
                            diff_unclip_embs = (base_unclip_embs[bs_idx].squeeze(dim=1).to(self.DEVICE).to(torch.float32) - pred_unclip_embs.squeeze(dim=1))

                            # CosineEmbeddingLoss between difference
                            movi_cos_loss = self.cos_loss(diff_unclip_embs.T,
                                                    diff_img_embs.T.to(self.DEVICE), self.target).half()  #
                            # control NAN INF in loss
                            if torch.isnan(movi_cos_loss).sum() or torch.isinf(movi_cos_loss).sum():
                              print(f"\rMovi {id_movi} movi_cos_loss_base isnan or isinf")
                              self.logs["naninf_cos"].append(id_movi)
                              break

                            # MSELoss between predict and
                            movi_mse_loss = self.mse_loss(diff_unclip_embs.half(),
                                                    diff_img_embs.half().to(self.DEVICE)).mean(0)#
                            # control NAN INF in loss
                            if torch.isnan(movi_mse_loss).sum() or torch.isinf(movi_mse_loss).sum():
                              print(f"\rMovi {id_movi} movi_mse_loss_base isnan or isinf")
                              self.logs["naninf_mse"].append(id_movi)
                              break

                            # collect base_loss
                            movi_loss_base = movi_cos_loss + movi_mse_loss # (0) for 1280
                            eph_base_loss = movi_loss_base.mean().item()

                            # CosineEmbeddingLoss between difference
                            cos_acc = self.cos_loss(pred_unclip_embs.squeeze(1).T,
                                              img_embs[bs_idx].squeeze(1).T.to(self.DEVICE), (-1)*self.target).half()

                            # control NAN INF in acc
                            if torch.isnan(cos_acc).sum() or torch.isinf(cos_acc).sum():
                              print(f"\rMovi {id_movi} cos_acc_base isnan or isinf")
                              self.logs["naninf_acc"].append(id_movi)
                              break
                            # collect cos_acc
                            eph_cos_acc+= cos_acc.mean().item()

                            # temp show losses for batch
                            if id_m:
                              print(f"\rMovi {id_movi} step {id_m} base_way mse {eph_loss_mse/id_m:.5f} | cos {eph_loss_cos/id_m:.5f} | lr {cur_lr:.5e}", end="")

                            # clear
                            del(time_base, directions, diff_img_embs, diff_unclip_embs)
                            self.flush_memory()

                            #########  Rotation train steps ########################################
                            to_rote, rote_norm, rote_back = 0, 0, 0
                            if self.add_rote_train:# and (epoch - self.last_epoch):
                                # Rotation train steps
                                rote_norm = len(config_norm['id_uclip_emb'])
                                rote_back = len(config_back['id_uclip_emb'])

                                # control size batch
                                if rote_norm > self.max_batch -1:
                                  config_norm['id_uclip_emb'] = config_norm['id_uclip_emb'][:self.max_batch - 1]
                                  config_norm['id_img_emb_s'] = config_norm['id_img_emb_s'][:self.max_batch - 1]
                                  config_norm['id_img_delta'] = config_norm['id_img_delta'][:self.max_batch - 1]
                                  config_norm['delta'] = config_norm['delta'][:self.max_batch - 1]
                                  rote_norm = len(config_norm['id_uclip_emb'])

                                if rote_back > self.max_batch -1:
                                  config_back['id_uclip_emb'] = config_back['id_uclip_emb'][:self.max_batch - 1]
                                  config_back['id_img_emb_s'] = config_back['id_img_emb_s'][:self.max_batch - 1]
                                  config_back['id_img_delta'] = config_back['id_img_delta'][:self.max_batch - 1]
                                  config_back['delta'] = config_back['delta'][:self.max_batch - 1]
                                  rote_back = len(config_back['id_uclip_emb'])

                                to_rote =  rote_norm + rote_back

                            if self.add_rote_train and to_rote: # and (epoch - self.last_epoch)
                                text_hid_states_2rt = []
                                unclip_embs_2rt = []
                                base_img_embs_2rt = []
                                image_embs_2rt = []
                                delta_2rt = []
                                direction_2rt = []

                                # intit class for shufflee again
                                srs = Shuff_Reshuff(to_rote)
                                if self.suff_direct:
                                    s_idx = srs.shuffle()   # shuffleed indexees
                                else:
                                    s_idx = srs.idx_base # shuffleed indexees


                                if rote_norm:
                                    # collect norm steps
                                    text_hid_states_2rt.append(torch.clone(text_hid_states[:d_batch])[config_norm['id_uclip_emb']])
                                    unclip_embs_2rt.append(torch.clone(base_unclip_embs[:d_batch])[config_norm['id_uclip_emb']])
                                    base_img_embs_2rt.append(torch.clone(img_embs[:d_batch])[config_norm['id_img_emb_s']])
                                    image_embs_2rt.append(torch.clone(img_embs[:d_batch])[config_norm['id_img_delta']])
                                    delta_2rt.append(torch.tensor(config_norm['delta']).unsqueeze(1))
                                    direction_2rt.append(torch.zeros_like(delta_2rt[-1]))


                                if rote_back and self.add_back_train:
                                    # collect back steps
                                    text_hid_states_2rt.append(torch.clone(text_hid_states[d_batch:])[config_back['id_uclip_emb']])
                                    unclip_embs_2rt.append(torch.clone(base_unclip_embs[d_batch:])[config_back['id_uclip_emb']])
                                    base_img_embs_2rt.append(torch.clone(img_embs[d_batch:])[config_back['id_img_emb_s']])
                                    image_embs_2rt.append(torch.clone(img_embs[d_batch:])[config_back['id_img_delta']])
                                    delta_2rt.append(torch.tensor(config_back['delta']).unsqueeze(1))
                                    direction_2rt.append(torch.ones_like(delta_2rt[-1]))


                                # shufle
                                text_hid_states_2rt = torch.concat(text_hid_states_2rt)[s_idx].to(self.DEVICE).to(torch.float32)
                                unclip_embs_2rt = torch.concat(unclip_embs_2rt)[s_idx].to(self.DEVICE).to(torch.float32)
                                base_img_embs_2rt = torch.concat(base_img_embs_2rt)[s_idx].to(self.DEVICE).to(torch.float32)
                                image_embs_2rt = torch.concat(image_embs_2rt)[s_idx].to(self.DEVICE).to(torch.float32)
                                delta_2rt = torch.concat(delta_2rt)[s_idx][s_idx].to(self.DEVICE)
                                direction_2rt = torch.concat(direction_2rt)[s_idx].to(self.DEVICE)

                                # get cos_sim  base_img_embs vectors and base_unclip_embs vectors
                                cos_sim = torch.cosine_similarity(base_img_embs[s_idx].to(self.DEVICE), base_unclip_embs[s_idx], dim = 2)

                                # get rotation marixes_i2i
                                R_marixes_i2i = self.RV.get_rotation_matrix(base_img_embs[s_idx].squeeze(1).to(torch.float32).to(self.DEVICE),
                                                      base_img_embs_2rt.squeeze(1).to(torch.float32)).to(self.DEVICE)

                                # compute roted unclip_embs with R_marixes_i2i and cos_sim base_img_embs and base_unclip_embs
                                unclip_embs_2rt = self.RV.cosim_rotate(unclip_embs_2rt, cos_sim.to(self.DEVICE), R_marixes_i2i)

                                # get rotation marixes_u2u
                                R_marixes_u2u = self.RV.get_rotation_matrix(base_unclip_embs[s_idx].squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                      unclip_embs_2rt.squeeze(1).to(torch.float32))

                                # compute roted text_hid_states with R_marixes_u2u and cos_sim base_img_embs and base_unclip_embs
                                text_hid_states_2rt = text_hid_states_2rt @ R_marixes_u2u


                                # rotation predict
                                pred_rote_embs = self.model(
                                              text_hidden_states = text_hid_states_2rt,
                                              prior_embeds = unclip_embs_2rt,
                                              delta_time = delta_2rt,
                                              direction = direction_2rt
                                                                )
                                # difference
                                diff_unclip_embeds = (unclip_embs_2rt.squeeze(dim=1) - pred_rote_embs.squeeze(dim=1))
                                diff_img_embs = (base_img_embs_2rt.squeeze(dim=1) - image_embs_2rt.squeeze(dim=1))

                                # CosineEmbeddingLoss between difference
                                movi_cos_loss += self.cos_loss(diff_unclip_embeds.T,
                                                                          diff_img_embs.T.to(self.DEVICE), self.target).half()  #
                                # control NAN INF in loss
                                if torch.isnan(movi_cos_loss).sum() or torch.isinf(movi_cos_loss).sum():
                                  print(f"\rMovi {id_movi} movi_cos_loss_rote isnan or isinf")
                                  self.logs["naninf_cos"].append(id_movi)
                                  break

                                # MSELoss between predict and
                                movi_mse_loss += self.mse_loss(diff_unclip_embeds.half(),
                                                                          diff_img_embs.half().to(self.DEVICE)).mean(0) #
                                # control NAN INF in loss
                                if torch.isnan(movi_mse_loss).sum() or torch.isinf(movi_mse_loss).sum():
                                  print(f"\rMovi {id_movi} movi_mse_loss_rote isnan or isinf")
                                  self.logs["naninf_mse"].append(id_movi)
                                  break

                                del(delta_2rt, direction_2rt, diff_img_embs, diff_unclip_embeds)
                                del(pred_rote_embs, text_hid_states_2rt, R_marixes_i2i, R_marixes_u2u)
                                del(unclip_embs_2rt,image_embs_2rt)
                                self.flush_memory()

                            #########  Diff train steps ########################################
                            to_diff = 0
                            # Diff train steps
                            if self.add_diff_train:
                                config_diff_norm, config_diff_back = self.maker_points.getpoints_diftrain()
                                diff_norm = len(config_diff_norm['id_uclip_emb'])
                                diff_back = len(config_diff_back['id_uclip_emb'])
                                to_diff = diff_norm + diff_back

                            if self.add_diff_train  and to_diff: # and (epoch - self.last_epoch)>1
                                if self.window:
                                    diff_loss_weight = 1- np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])
                                else:
                                    diff_loss_weight = 1 - self.hist["base_loss"][-1]/self.hist["base_loss"][0]
                                if diff_loss_weight<0: diff_loss_weight = 0

                            if self.add_diff_train and to_diff and diff_loss_weight: # and (epoch - self.last_epoch) >1
                                take_text_hid_states = []
                                take_base_unclip_embs = []
                                next_unclip_embs = []
                                next_base_img_embs = []
                                next_image_embs = []
                                next_delta = []
                                next_text_hid_states = []
                                next_direction = []

                                # un_shuffleed
                                pred_unclip_embs = torch.clone(pred_unclip_embs.detach().cpu())[bu_idx]

                                # intit class for shufflee again
                                srs = Shuff_Reshuff(to_diff)
                                if self.suff_direct:
                                    s_idx = srs.shuffle() # shuffleed indexees
                                else:
                                    s_idx = srs.idx_base # shuffleed indexees

                                if diff_norm:
                                    # collect diff norm steps
                                    take_base_unclip_embs.append(torch.clone(base_unclip_embs[:d_batch])[config_diff_norm['id_uclip_emb']])
                                    take_text_hid_states.append(torch.clone(text_hid_states[:d_batch])[config_diff_norm['id_uclip_emb']])
                                    next_unclip_embs.append(torch.clone(pred_unclip_embs[:d_batch])[config_diff_norm['id_uclip_emb']])
                                    next_base_img_embs.append(torch.clone(img_embs[:d_batch])[config_diff_norm['id_img_emb_s']])
                                    next_image_embs.append(torch.clone(img_embs[:d_batch])[config_diff_norm['id_img_delta']])
                                    next_delta.append(torch.tensor(config_diff_norm['delta']).unsqueeze(1))
                                    next_direction.append(torch.zeros_like(next_delta[-1]))

                                if diff_back:
                                    # collect diff back steps
                                    take_base_unclip_embs.append(torch.clone(base_unclip_embs[:d_batch])[config_diff_back['id_uclip_emb']])
                                    take_text_hid_states.append(torch.clone(text_hid_states[d_batch:])[config_diff_back['id_uclip_emb']])
                                    next_unclip_embs.append(torch.clone(pred_unclip_embs[d_batch:])[config_diff_back['id_uclip_emb']])
                                    next_base_img_embs.append(torch.clone(img_embs[d_batch:])[config_diff_back['id_img_emb_s']])
                                    next_image_embs.append(torch.clone(img_embs[d_batch:])[config_diff_back['id_img_delta']])
                                    next_delta.append(torch.tensor(config_diff_back['delta']).unsqueeze(1))
                                    next_direction.append(torch.ones_like(next_delta[-1]))


                                # collect torch.concat
                                take_base_unclip_embs = torch.concat(take_base_unclip_embs).to(self.DEVICE)
                                take_text_hid_states = torch.concat(take_text_hid_states).to(self.DEVICE)
                                next_unclip_embs = torch.concat(next_unclip_embs).to(self.DEVICE).to(torch.float32)
                                next_base_img_embs = torch.concat(next_base_img_embs).to(torch.float32)
                                next_image_embs = torch.concat(next_image_embs).to(torch.float32)
                                next_delta = torch.concat(next_delta)[s_idx].to(self.DEVICE)
                                next_direction = torch.concat(next_direction)[s_idx].to(self.DEVICE)

                                # get rotation vectors
                                R_marixes = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1).to(torch.float32),
                                                                        next_unclip_embs.squeeze(1))
                                next_text_hid_states = take_text_hid_states.to(torch.float32) @ R_marixes

                                # dif predict from base predict
                                next_pred_unclip_embs = self.model(
                                    text_hidden_states = next_text_hid_states[s_idx],
                                    prior_embeds = next_unclip_embs[s_idx],
                                    delta_time = next_delta,
                                    direction = next_direction
                                    )

                                # difference
                                diff_unclip_embeds = (next_unclip_embs[s_idx].squeeze(dim=1).to(self.DEVICE) - next_pred_unclip_embs.squeeze(dim=1))
                                diff_img_embs =  (next_base_img_embs[s_idx].squeeze(dim=1) - next_image_embs[s_idx].squeeze(dim=1))

                                # CosineEmbeddingLoss between difference
                                movi_cos_loss += self.cos_loss(diff_unclip_embeds.T,
                                                                            diff_img_embs.T.to(self.DEVICE), self.target).half()  #
                                # control NAN INF in loss
                                if torch.isnan(movi_cos_loss).sum() or torch.isinf(movi_cos_loss).sum():
                                  print(f"\rMovi {id_movi} movi_cos_loss_diff isnan or isinf")
                                  self.logs["naninf_cos"].append(id_movi)
                                  break

                                # MSELoss between predict and
                                movi_mse_loss +=  self.mse_loss(diff_unclip_embeds.half(),
                                                                            diff_img_embs.half().to(self.DEVICE)).mean(0) #
                                # control NAN INF in loss
                                if torch.isnan(movi_mse_loss).sum() or torch.isinf(movi_mse_loss).sum():
                                  print(f"\rMovi {id_movi} movi_mse_loss_diff isnan or isinf")
                                  self.logs["naninf_mse"].append(id_movi)
                                  break

                                del(diff_img_embs, diff_unclip_embeds)
                                del(next_image_embs, next_base_img_embs, next_delta, next_direction)
                                del(next_text_hid_states,  next_unclip_embs)
                                del(pred_unclip_embs, next_pred_unclip_embs)
                                self.flush_memory()


                            # collect loss
                            movi_loss = movi_cos_loss + movi_mse_loss # (0) for 1280
                            movi_loss.backward(torch.ones_like(movi_loss))
                            print(f"\rMovi_loss.backward on {id_m} step ", end="")

                            eph_loss_mse += movi_mse_loss.mean().item()
                            eph_loss_cos += movi_cos_loss.mean().item()
                            eph_loss += movi_loss.mean().item()

                            # make clip_grad_norm model
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()

                            # flush_memory
                            del(movi_cos_loss, movi_mse_loss)
                            del(image_embeds, base_img_embs, img_embs)
                            del(text_hid_states, base_unclip_embs)
                            self.flush_memory()

                        else:

                          self.logs["bad_movi"].append(id_movi)
                          print(f'\rMovi {id_movi} has some static frames', end="")

                    else:
                      print(f'Movi {id_movi} empty')
                      self.logs["empty"].append(id_movi)

                # collect data
                good_steps = len(take_data)
                eph_cos_acc/=good_steps
                eph_loss/=good_steps
                eph_loss_mse/=good_steps
                eph_loss_cos/=good_steps
                eph_base_loss/=good_steps

                self.hist["lr"].append(cur_lr)
                self.hist["loss"].append(eph_loss)
                self.hist["loss_mse"].append(eph_loss_mse)
                self.hist["loss_cos"].append(eph_loss_cos)
                self.hist["acc"].append(eph_cos_acc)
                self.hist["base_loss"].append(eph_base_loss)

                self.scheduler.step()

                 # compute av_weighted losses and acc by window
                if self.window:
                    av_acc =   np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])
                    av_mse =   np.mean(self.hist["loss_mse"][-min(len(self.hist["loss_mse"]), self.window):])
                    av_cos =   np.mean(self.hist["loss_cos"][-min(len(self.hist["loss_cos"]), self.window):])
                    av_loss =   np.mean(self.hist["loss"][-min(len(self.hist["loss"]), self.window):])

                else:
                    av_acc =  self.hist["acc"][-1]
                    av_mse =  self.hist["loss_mse"][-1]
                    av_cos =  self.hist["loss_cos"][-1]
                    av_loss =  self.hist["loss"][-1]

                if epoch - self.last_epoch > self.update_best:

                    if self.best_acc < av_acc:
                        self.best_acc = av_acc
                        self.last_save = "acc"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_mse > av_mse:
                        self.best_loss_mse = av_mse
                        self.last_save = "mse"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_cos > av_cos:
                        self.best_loss_cos = av_cos
                        self.last_save = "cos"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss > av_loss:
                        self.best_loss = av_loss
                        self.last_save = "loss"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    # same pereodicaly station model to check
                    if (epoch - self.last_epoch) and not epoch % self.fr_sv_cpt:
                        text += f'- save_{epoch}_ep_cpt '
                        model_name = f"/tmp_{epoch}_a_{av_acc:.3f}_l_{av_loss:.3f}_c_{av_cos:.3f}_m_{av_mse:.3f}.pt"
                        torch.save(self.model.state_dict(), self.dict_model_paths["temp"] + model_name)

                    if not save_model:
                        self.wait_train+=1
                        text += f'wait_{self.wait_train}'
                    else:
                      self.wait_train = 0 - self.window
                      self.wait2end = 0

                # finish training with loading best to predict
                if self.wait_train  > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1 or not epoch % self.fr_sv_cpt:

                    if self.wait_train > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1:
                        # load last best state
                        self.model.load_state_dict(torch.load(self.dict_model_paths[self.last_save]))
                        text += f' - load best_{self.last_save}_model {self.best_eph[self.last_save]} ep'

                    if self.wait_train > self.JDUN_TRAIN:
                        # update scheduler and optimizer
                        self.GAMMA *= self.EXP_RATE
                        self.LR_RATE =  self.hist["lr"][-1]*self.EXP_RATE
                        for g in self.optimizer.param_groups:
                            g['lr'] = self.LR_RATE
                        self.scheduler.gamma=self.GAMMA
                        self.scheduler.last_epoch = epoch
                        self.wait_train = 0
                        self.wait2end += 1


                    if self.wait2end > self.JDUN_END or epoch == self.last_epoch + self.EPOCHS -1 or not epoch % self.fr_sv_cpt:
                        if epoch and not epoch % self.fr_sv_cpt:
                            text += f' - save best_{self.last_save}_model {self.best_eph[self.last_save]} ep checkpoint'
                        if (epoch - self.last_epoch):
                            # take hist before best
                            self.hist = {key: self.hist[key][:self.best_eph[self.last_save]+1] for key in self.hist.keys()}
                            # update last history
                            update_dict_hist = {key: self.history_train[key] + self.hist[key] for key in self.history_train.keys()}

                            name_model = str(self.model_class) if not self.change_name_to_save else str(self.change_name_to_save)
                            # save checkpoint
                            torch.save({
                                'model_class' : self.model_class,
                                'saved_model' : self.last_save,
                                'epoch': self.best_eph[self.last_save],
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.best_loss,
                                'cos_loss': self.best_loss_cos,
                                'mse_loss': self.best_loss_mse,
                                'acc': self.best_acc,
                                'last_lr': self.hist["lr"][-1],
                                'last_checkpoint' : self.last_checkpoint,
                                'last_history': self.hist,
                                'all_history': update_dict_hist,
                                }, self.path_save + f'/{name_model}_{self.last_save}_{self.best_eph[self.last_save]}.cpt')

                        if (epoch - self.last_epoch) and self.wait2end > self.JDUN_END:
                            print(f"\nStop train, don't good fitness already on {epoch} ep, save best model from {self.best_eph[self.last_save]} ep")
                            break

                        if epoch == self.last_epoch + self.EPOCHS -1:
                            print(f'\nTrain is finished, saved the best model from {self.best_eph[self.last_save]}_ep')

                print(f'\rEp {epoch} all_loss {av_loss:.5f} | acc {av_acc:.5f} | mse_loss {av_mse:.5f} | cos_loss {av_cos:.5f} | lr {cur_lr:.5e} {text}\n')
        ###
        def flush_memory(self):
              import gc
              gc.collect()
              torch.cuda.empty_cache()
              torch.cuda.ipc_collect()
              with torch.no_grad():
                  for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()