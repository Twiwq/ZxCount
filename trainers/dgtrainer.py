import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from trainers.trainer import Trainer
from utils.misc import denormalize, divide_img_into_patches, my_fft, decoder_image



class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets = targets.view(targets.size(0), -1)
        loss = self.loss(predictions, targets)
        return loss




def convert_to_small_scale(den, patch_size=8):

    pool_filter = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size).cuda()


    target = pool_filter(den)
    total_sum = target * (patch_size * patch_size)
    return total_sum


def den2cls(gt_den_map, label_bd):
    B, C, H, W = gt_den_map.shape
    d_map = gt_den_map.flatten(1).unsqueeze(2)
    label_bd = label_bd.unsqueeze(0).unsqueeze(0)
    cls_map = ((d_map - label_bd) > 0).sum(dim=-1)
    cls_map = cls_map.reshape(B,C, H, W)
    return cls_map

class DGTrainer(Trainer):
    def __init__(self, seed, version, device, log_para, patch_size, mode):
        super().__init__(seed, version, device)

        self.log_para = log_para
        self.patch_size = patch_size
        self.mode = mode
        self.loss_fn_MSE = torch.nn.MSELoss()
        self.cls_loss = SegmentationLoss().to(self.device)

        label_count_patch = torch.tensor(
            [0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209,
             0.03357841819524765 ,
             0.04810526967048645 , 0.06570728123188019 , 0.08683456480503082, 0.11207923293113708,
             0.1422334909439087 ,
             0.17838051915168762 , 0.22167329490184784, 0.2732916474342346, 0.33556100726127625,
             0.41080838441848755 ,
             0.5030269622802734 , 0.6174761652946472 , 0.762194037437439, 0.9506691694259644,
             1.2056223154067993 ,
             1.5706151723861694 , 2.138580322265625, 3.233219861984253, 7.914860725402832])   #7.914860725402832



        self.label_count_patch = label_count_patch.unsqueeze(0).unsqueeze(2).to(self.device)  # (1, 24, 1)


    def load_ckpt(self, model, path):
        if isinstance(model, list):
            if path is not None:
                super().load_ckpt(model[0], path[0])
                super().load_ckpt(model[1], path[1])
        else:
            super().load_ckpt(model, path)

    def save_ckpt(self, model, path):
        if isinstance(model, list):
            super().save_ckpt(model[0], path.replace('.pth', '_gen.pth'))
            super().save_ckpt(model[1], path.replace('.pth', '_reg.pth'))
        else:
            super().save_ckpt(model, path)



    def simi_loss(self, bg_feature,fg_feature, memory_list, gt_bool, gaum):
        b, c, h, w = bg_feature.shape  # (16, 256, 80, 80)
        bg_feature_reshaped = bg_feature.view(b, c, -1).permute(0, 2, 1)  # (16, 1600, 256)
        fg_feature_reshaped = fg_feature.view(b, c, -1).permute(0, 2, 1)  # (16, 6400, 256)
        batch_idx_bg, pixel_idx_bg = torch.where(~gt_bool)
        bg = bg_feature_reshaped[batch_idx_bg, pixel_idx_bg]  # (N_fg, 256)
        bg_sum = torch.norm(bg, dim=-1, keepdim=True) + 1e-5  # (16, 6400,1)
        memory_sum = torch.norm(memory_list, dim=-1, keepdim=True) + 1e-5  # (24，1)
        simi_bg = torch.matmul(bg, memory_list.T)
        simi_bg = simi_bg / (bg_sum * memory_sum.T)
        loss_simi1 = torch.mean(torch.mean(simi_bg+ 1, dim=-1))

        if gt_bool.any():
            batch_idx, pixel_idx = torch.where(gt_bool)
            fg = fg_feature_reshaped[batch_idx, pixel_idx]  # (N_fg, 256)
            fg_sum = torch.norm(fg, dim=-1, keepdim=True) + 1e-5  # (16, 6400,1)
            simi_fg = torch.matmul(fg, memory_list.T)
            simi_fg = simi_fg / (fg_sum * memory_sum.T)


            gaum_filtered = torch.where(gaum > 0, gaum - 1, gaum)

            gaum_fg = gaum_filtered[batch_idx, pixel_idx]

            match_value = simi_fg.gather(1, gaum_fg.unsqueeze(1))

            loss_simi2 = torch.mean(1 - match_value.squeeze())


            level_diff = torch.abs(
                gaum_fg.unsqueeze(1) - torch.arange(memory_list.size(0), device=fg_feature_reshaped.device))
            rebalanced = 2 * torch.exp(-level_diff / 0.5) - 1
            positive = rebalanced > 0
            rebalanced = torch.abs(rebalanced)


            memory_detached = memory_list.detach()
            simi_de = torch.matmul(fg, memory_detached.T)
            simi_de = simi_de/((fg_sum * memory_sum.T.detach()))

            logits = simi_de / 0.1
            logits_max = torch.max(logits, dim=1, keepdim=True)[0]
            stable_logits = logits - logits_max.detach()

            exp_logits = torch.exp(stable_logits)
            weighted_exp = exp_logits * rebalanced.detach()


            numerator = torch.sum(weighted_exp * positive.float(), dim=1)
            denominator = torch.sum(weighted_exp, dim=1) + 1e-6
            soft_loss = numerator / denominator
            loss_simi3 = torch.mean(-torch.log(soft_loss + 1e-6))
            loss_simi =  loss_simi1+loss_simi2 + loss_simi3
        else:

            simi_de = torch.empty((0, memory_list.size(0)), device=self.device)
            loss_simi = loss_simi1

        return loss_simi, F.softmax(simi_fg,dim=-1),F.softmax(simi_de,dim=-1)



    def Scl(self, glo_fea, pure_fea,memory_list_glo, memory_list_mask, gt_bool):
        b, c, h, w = glo_fea.shape  # (16, 256, 80, 80)
        glo_fea_reshaped = glo_fea.view(b, c, -1).permute(0, 2, 1)  # (16, 1600, 256)
        pure_fea_reshaped = pure_fea.view(b, c, -1).permute(0, 2, 1)  # (16, 1600, 256)

        batch_idx, pixel_idx = torch.where(gt_bool)
        glo_fg = glo_fea_reshaped[batch_idx, pixel_idx]  # (N_fg, 256)
        pure_fg = pure_fea_reshaped[batch_idx, pixel_idx]  # (N_fg, 256)


        glo_mem_sum = torch.norm(memory_list_glo, dim=-1, keepdim=True) + 1e-5  # (24，1)
        pure_mem_sum = torch.norm(memory_list_mask, dim=-1, keepdim=True) + 1e-5  # (24，1)

        glo_fg_sum = torch.norm(glo_fg, dim=-1, keepdim=True) + 1e-5  # (16, 6400,1)
        pure_fg_sum = torch.norm(pure_fg, dim=-1, keepdim=True) + 1e-5  # (16, 6400,1)


        glo_score1 = torch.matmul(glo_fg, memory_list_glo.T)
        glo_score1 = glo_score1 / (glo_fg_sum * glo_mem_sum.T)
        glo_score1 = F.softmax(glo_score1,dim=-1)

        pure_score1 = torch.matmul(pure_fg, memory_list_glo.T.detach())
        pure_score1 = pure_score1 / (pure_fg_sum * glo_mem_sum.T.detach())
        pure_score1 = F.softmax(pure_score1,dim=-1)

        loss_con1=F.mse_loss(glo_score1.detach(),pure_score1)


        return loss_con1



    def compute_count_loss(self, loss: nn.Module, pred_dmaps, gt_datas, weights=None):
        if loss.__class__.__name__ == 'MSELoss':
            _, gt_dmaps, _ = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            if weights is not None:
                pred_dmaps = pred_dmaps * weights
                gt_dmaps = gt_dmaps * weights
            loss_value = loss(pred_dmaps, gt_dmaps * self.log_para)

        elif loss.__class__.__name__ == 'BL':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            targs = [targ.to(self.device) for targ in targs]
            st_sizes = st_sizes.to(self.device)
            loss_value = loss(gts, st_sizes, targs, pred_dmaps)

        else:
            raise ValueError('Unknown loss: {}'.format(loss))

        return loss_value

    def compute_patch_loss(self, loss: nn.Module, pred_dmaps, den_patch, weights=None):
        if loss.__class__.__name__ == 'MSELoss':
            if weights is not None:
                pred_dmaps = pred_dmaps * weights
                gt_dmaps = gt_dmaps * weights
            loss_value = loss(pred_dmaps, den_patch * self.log_para)
        else:
            raise ValueError('Unknown loss: {}'.format(loss))

        return loss_value

    def predict(self, model, img):

        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)
            for patch in img_patches:
                pred = model(patch) if self.mode == 'base' else model(patch)[0]
                pred_count += torch.sum(pred).cpu().item() / self.log_para
        else:
            pred_dmap = model(img) if self.mode == 'base' else model(img)[0]
            pred_count = pred_dmap.sum().cpu().item() / self.log_para

        return pred_count

    def predict_isw(self, model, img, img2):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)
            img_patches2, _, _ = divide_img_into_patches(img2, ps)
            for patch, patch2 in zip(img_patches, img_patches2):
                pred = model(patch) if self.mode == 'base' else model(patch)[0]
                pred_count += torch.sum(pred).cpu().item() / self.log_para
                model([patch, patch2], cal_covstat=True)
        else:
            pred_dmap = model(img) if self.mode == 'base' else model(img)[0]
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            model([img, img2], cal_covstat=True)

        return pred_count

    def get_visualized_results(self, model, img):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            dmap = torch.zeros(1, 1, h, w)
            img_patches, nh, nw = divide_img_into_patches(img, ps)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i * nw + j]
                    pred_dmap = model(patch)
                    dmap[:, :, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = pred_dmap
        else:
            dmap = model(img)

        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()

        return dmap

    def get_visualized_results_with_cls(self, model, img):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            dmap = torch.zeros(1, 1, h, w)
            cmap = torch.zeros(1, 3, h // 16, w // 16)
            img_patches, nh, nw = divide_img_into_patches(img, ps)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i * nw + j]
                    pred_dmap, pred_cmap = model(patch)
                    dmap[:, :, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = pred_dmap
                    cmap[:, :, i * ps // 16:(i + 1) * ps // 16, j * ps // 16:(j + 1) * ps // 16] = pred_cmap
        else:
            dmap, cmap = model(img)

        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()
        cmap = cmap[0, 0].cpu().detach().numpy().squeeze()

        return dmap, cmap

    def train_step(self, model, loss, optimizer, batch, epoch):
        imgs1, imgs2, (h_imgs1, h_imgs2), (l_imgs1, l_imgs2), gt_datas = batch


        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)

        h_imgs1 = h_imgs1.to(self.device)
        h_imgs2 = h_imgs2.to(self.device)
        l_imgs1 = l_imgs1.to(self.device)
        l_imgs2 = l_imgs2.to(self.device)

        _, gt_dmaps, gt_cmaps = gt_datas
        gt_cmaps = gt_cmaps.to(self.device)
        gt_dmaps = gt_dmaps.to(self.device)  # torch.Size([16, 1, 320, 320])
        b, c, h, w = gt_dmaps.shape  # batch size
        # label_count_mask = self.label_count_mask.repeat(b, 1, 1)  # (16, 24, 1)
        label_count_patch = self.label_count_patch.repeat(b, 1, 1)  # (16, 24, 1)
        # gt_mask = self.generate_gt_mask(gt_dmaps, self.label_count_mask)

        if self.mode == 'base':
            optimizer.zero_grad()
            dmaps1 = model(imgs1)
            dmaps2 = model(imgs2)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_total = loss_den
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'add':
            optimizer.zero_grad()
            dmaps1, dmaps2, loss_con = model.forward_train(imgs1, imgs2)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_total = loss_den + loss_con
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'cls':
            optimizer.zero_grad()

            dmaps1, dmaps2, (cmaps1, cmaps2), (re_low1, re_low2), (re_high1, re_high2), interact_loss, (dh1, dh2), (
                cl1, cl2) = model(imgs1, imgs2, gt_cmaps)
            # re_loss
            relow_loss = self.loss_fn_MSE(re_low1, l_imgs1) + self.loss_fn_MSE(re_low2, l_imgs2)
            rehigh_loss = self.loss_fn_MSE(re_high1, h_imgs1) + self.loss_fn_MSE(re_high2, h_imgs2)
            re_loss = rehigh_loss + relow_loss

            # den_loss
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_den1 = self.compute_count_loss(loss, dh1, gt_datas) + self.compute_count_loss(loss, dh2, gt_datas)
            # cls_loss
            loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
            loss_cls1 = F.binary_cross_entropy(cl1, gt_cmaps) + F.binary_cross_entropy(cl2, gt_cmaps)

            loss_total = loss_den + 10 * loss_cls + 10 * re_loss + 10 * interact_loss + 0.5 * +loss_den1 + 5 * loss_cls1
            loss_total.backward()
            optimizer.step()


        elif self.mode == 'final_single':
            optimizer.zero_grad()
            gt_dmaps = convert_to_small_scale(gt_dmaps)
            gau = gt_dmaps.view(b, -1).unsqueeze(1) - label_count_patch

            gaum = torch.sum(gau > 0, dim=1)
            gt_bool = gaum.bool()

            # gt_cmaps=F.interpolate(gt_cmaps, scale_factor=4, mode='bilinear')
            dmaps1, pred_m1, feature1, pure_feature1 = model.forward_train(imgs1, gt_mask)
            simi_loss1, simi_ori = self.simi_loss(feature1, pure_feature1, model.memory_list, gt_bool, gaum)

            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas)
            loss_mask = F.binary_cross_entropy(pred_m1, gt_mask)

            loss_total = loss_den + simi_loss1 + 10 * loss_mask

            loss_total.backward()
            optimizer.step()

        elif self.mode == 'final':
            optimizer.zero_grad()

            gt_dmaps_patch = convert_to_small_scale(gt_dmaps)


            gau = gt_dmaps_patch.view(b, -1).unsqueeze(1) - label_count_patch
            gaum = torch.sum(gau > 0, dim=1)
            gaum_re=gaum.reshape(b,1,40,40)


            gt_bool = gaum.bool()



            (dmaps1, dmaps2), (glo_down1,glo_down2,fg_down1,fg_down2,bg_down1,bg_down2),l_err,(c1,c2),(cls_score_fg1, cls_score_fg2),(fg_dp1,fg_dp2) = model.forward_train(imgs1, imgs2, gt_cmaps)


            siyi_loss1, simi_ori_fg, ori_de = self.simi_loss(glo_down1,glo_down1,model.memory_list_global, gt_bool, gaum)
            simi_loss2, simi_aug_fg, aug_de = self.simi_loss(glo_down2,glo_down2,model.memory_list_global, gt_bool, gaum)


            simi_loss_mask1, simi_ori_mask,mask_de_ori = self.simi_loss(bg_down1,fg_down1,model.memory_list_mask, gt_bool, gaum)
            simi_loss_mask2, simi_aug_mask,mask_de_aug = self.simi_loss(bg_down2,fg_down2,model.memory_list_mask, gt_bool, gaum)


            loss_con = F.mse_loss(ori_de, aug_de) + F.mse_loss(simi_ori_mask, simi_aug_mask)


            loss_sim = (simi_loss1 + simi_loss2) + \
                        simi_loss_mask1 + simi_loss_mask2


            loss_cls_score = self.cls_loss(cls_score_fg1, gaum_re).mean()+self.cls_loss(cls_score_fg2, gaum_re).mean()

            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)



            loss_cls = F.binary_cross_entropy(c1, gt_cmaps) + F.binary_cross_entropy(c2, gt_cmaps)

            loss_total = loss_den +1*loss_sim+10*loss_cls + 10*loss_cls_score+10*l_err +  10* loss_con



            loss_total.backward()
            optimizer.step()


        elif self.mode == 'isw':
            optimizer.zero_grad()
            gts = gt_datas[1].to(self.device)
            losses = model(imgs1, gts=gts, apply_wtloss=(epoch > 5))
            loss_total = torch.FloatTensor([0]).cuda()
            loss_total += losses[0]
            # loss_total += 0.4 * losses[1]
            if epoch > 5:
                loss_total += 0.6 * losses[1]
            loss_total.backward()
            optimizer.step()

        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

        return loss_total.detach().item()

    def val_step(self, model, batch):

        img1, img2, gt, _, _ = batch
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'isw':
            with torch.no_grad():
                pred_count = self.predict_isw(model, img1, img2)
        else:
            pred_count = self.predict(model, img1)
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2

        return mae, {'mse': mse}

    def test_step(self, model, batch):
        img1, _, gt, _, _ = batch
        img1 = img1.to(self.device)
        # img2 = img2.to(self.device)

        pred_count = self.predict(model, img1)
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        return {'mae': mae, 'mse': mse}

    def vis_step(self, model, batch):
        img1, img2, gt, name, _ = batch
        vis_dir = os.path.join(self.log_dir, 'vis')
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'base':
            pred_dmap1 = self.get_visualized_results(model, img1)
            pred_dmap2 = self.get_visualized_results(model, img2)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            img2 = denormalize(img2.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count1 = pred_dmap1.sum() / self.log_para
            pred_count2 = pred_dmap2.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img1, pred_dmap1, img2, pred_dmap2]
            titles = [name, f'Pred1: {pred_count1}', f'GT: {gt_count}', f'Pred2: {pred_count2}']

            fig = plt.figure(figsize=(10, 6))
            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()

        else:
            pred_dmap1, pred_cmap1 = self.get_visualized_results_with_cls(model, img1)
            pred_dmap2, pred_cmap2 = self.get_visualized_results_with_cls(model, img2)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            img2 = denormalize(img2.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count1 = pred_dmap1.sum() / self.log_para
            pred_count2 = pred_dmap2.sum() / self.log_para
            gt_count = gt.shape[1]

            new_cmap1 = pred_cmap1.copy()
            new_cmap1[new_cmap1 < 0.5] = 0
            new_cmap1[new_cmap1 >= 0.5] = 1
            # new_cmap1 = np.resize(new_cmap1, (new_cmap1.shape[0]*16, new_cmap1.shape[1]*16))
            new_cmap2 = pred_cmap2.copy()
            new_cmap2[new_cmap2 < 0.5] = 0
            new_cmap2[new_cmap2 >= 0.5] = 1
            # new_cmap2 = np.resize(new_cmap2, (new_cmap2.shape[0]*16, new_cmap2.shape[1]*16))

            datas = [img1, pred_dmap1, pred_cmap1, img2, pred_dmap2, pred_cmap2]
            titles = [name[0], f'Pred1: {pred_count1}', 'Cls1', f'GT: {gt_count}', f'Pred2: {pred_count2}', 'Cls2']

            fig = plt.figure(figsize=(15, 6))
            for i in range(6):
                ax = fig.add_subplot(2, 3, i + 1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])
            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))

            new_datas = [img1, pred_cmap1, new_cmap1, pred_dmap1]
            new_titles = [f'{name[0]}', 'Cls', 'BCls', f'Pred_{pred_count1}']
            for i in range(len(new_datas)):
                plt.imsave(os.path.join(vis_dir, f'{name[0]}_{new_titles[i]}.png'), new_datas[i])

            plt.close()