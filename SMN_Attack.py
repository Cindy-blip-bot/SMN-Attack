import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import random
import time
from torchvision import transforms
import cv2
from transformers import AutoModel, AutoImageProcessor
import math
# ==================== Attacker ====================
class Attacker():
    def __init__(self, model, img_attacker, txt_attacker, target_model=None):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.target_model = target_model

    def attack(self, imgs, txts, txt2img, all_texts, device='cpu', max_length=30,
               scales=None, masks=None, epsilon=8/255, **kwargs):
        total_start = time.time()
        b = imgs.size(0)
        clean_imgs = imgs.clone().detach().to(device)
        clean_txts = list(txts)
        if torch.is_tensor(txt2img):
            txt2img_tensor = txt2img.to(device, dtype=torch.long)
        else:
            txt2img_tensor = torch.tensor(txt2img, device=device, dtype=torch.long)

        with torch.no_grad():
            all_texts_input = self.txt_attacker.tokenizer(
                all_texts, padding='max_length', truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)
            all_txt_supervisions = self.model.inference_text(all_texts_input)['text_feat']

            origin_img_output = self.model.inference_image(
                self.img_attacker.normalization(clean_imgs)
            )
            img_feat_orig = origin_img_output['image_feat']

        # Step 1: Text Attack
        img_supervisions_1 = img_feat_orig[txt2img_tensor]
        adv_txts_1 = self.txt_attacker.img_guided_attack(
            self.model, clean_txts, img_embeds=img_supervisions_1, beta_t=0.0
        )

        with torch.no_grad():
            txt_input_1 = self.txt_attacker.tokenizer(
                adv_txts_1, padding='max_length', truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)
            txt_feat_1_full = self.model.inference_text(txt_input_1)['text_feat']
            if self.txt_attacker.cls:
                txt_feat_1 = txt_feat_1_full[:, 0, :]
            else:
                txt_feat_1 = txt_feat_1_full.flatten(1)

        # Step 2: Image Attack
        adv_imgs_1, last_adv_imgs_1 = self.img_attacker.txt_guided_attack(
            self.model, clean_imgs, txt2img_tensor,
            all_txt_supervisions=all_txt_supervisions, device=device,
            scales=scales, txt_embeds=txt_feat_1
        )

        total_end = time.time()
        return adv_imgs_1, adv_txts_1, total_end - total_start

class ImageAttacker:
    def __init__(self, normalization=None, eps=2/255, steps=10, momentum=0.85,
                 eps_inner=8/255, eps_outer=2/255, use_crop=True,
                 crop_ratio_range=(0.8, 1.0), inner_ratio=0.60, num_primary_regions=2,
                 dino_model_name="/your/path/to/dino-vit", path_samples=3, path_sigma=0.01,
                 mask_update_interval=1):
        self.normalization = normalization
        self.eps, self.steps, self.momentum = eps, steps, momentum
        self.step_size_inner, self.step_size_outer = 2/255, 0.75/255
        self.eps_inner, self.eps_outer = eps_inner, eps_outer
        self.use_crop, self.crop_ratio_range = use_crop, crop_ratio_range
        self.inner_ratio, self.num_primary_regions = inner_ratio, num_primary_regions
        self.dino_model_name = dino_model_name
        self.path_samples, self.path_sigma = path_samples, path_sigma
        self.mask_update_interval = mask_update_interval
        self._dino_model = self._dino_processor = self._dino_device = None

    # ========== DINO mask ==========
    def _load_dino(self, device):
        device = torch.device(device)
        if self._dino_model is None or self._dino_device != device:
            self._dino_processor = AutoImageProcessor.from_pretrained(self.dino_model_name)
            self._dino_model = AutoModel.from_pretrained(self.dino_model_name).to(device).eval()
            self._dino_device = device
        return self._dino_model, self._dino_processor

    def get_dino_masks(self, imgs, device):
        dino_model, processor = self._load_dino(device)
        B, C, H, W = imgs.shape
        size_cfg = processor.size
        target_h = size_cfg.get("height", 224) if isinstance(size_cfg, dict) else (size_cfg or 224)
        target_w = size_cfg.get("width", target_h) if isinstance(size_cfg, dict) else target_h
        ps = getattr(dino_model.config, "patch_size", 16)
        target_h, target_w = max(ps, int(target_h // ps * ps)), max(ps, int(target_w // ps * ps))

        imgs_r = F.interpolate(imgs, size=(target_h, target_w), mode="bilinear", align_corners=False)
        inputs = processor(images=imgs_r, return_tensors="pt", do_rescale=False).to(device)
        with torch.no_grad():
            attn = dino_model(**inputs, output_attentions=True).attentions[-1]

        cls_attn = attn[:, :, 0, 1:].mean(dim=1).reshape(B, 1, target_h // ps, target_w // ps)
        attn = F.interpolate(cls_attn, size=(H, W), mode="bilinear", align_corners=False)
        mn, mx = attn.amin(dim=(2,3), keepdim=True), attn.amax(dim=(2,3), keepdim=True)
        raw = (attn - mn) / (mx - mn + 1e-8)
        thr = torch.quantile(raw.flatten(start_dim=1), 1 - self.inner_ratio, dim=1).view(B,1,1,1)
        m = (raw > thr).float()

        for b in range(B):
            m_np = m[b,0].cpu().numpy().astype(np.uint8)
            n, lb, st, _ = cv2.connectedComponentsWithStats(m_np, 8)
            if n > 1:
                keep = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:self.num_primary_regions] + 1
                m[b,0] = torch.from_numpy(np.isin(lb, keep).astype(np.float32)).to(m.device)

        inner = m
        if inner.size(1) == 1 and C != 1:
            inner = inner.repeat(1, C, 1, 1)
        return inner, raw

    def get_image_feat(self, model, imgs):
        if self.normalization is not None:
            imgs = self.normalization(imgs)
        output = model.inference_image(imgs)
        if isinstance(output, dict):
            return output.get("image_feat") if output.get("image_feat") is not None else output.get("img_embeds")
        return output

    def loss_func(self, embeds, txt_embeds, txt2img, proj):
        B, N = embeds.shape[0], txt_embeds.shape[0]
        dev = embeds.device
        sim_proj = embeds @ proj @ (txt_embeds @ proj).T
        labels = torch.zeros(B, N, device=dev)
        labels[txt2img, torch.arange(N, device=dev)] = 1
        loss1 = -(sim_proj * labels).sum(-1).mean()

        sim = embeds @ txt_embeds.T
        correct = sim[txt2img, torch.arange(N, device=dev)]
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask[txt2img, torch.arange(N, device=dev)] = False
        hard = (sim * mask.float()).max(dim=1)[0]
        loss2 = -correct.mean() + 0.5 * hard.mean()
        return 0.4 * loss1 + 0.6 * loss2

    def get_scaled_imgs(self, imgs, scales, mask=None):
        
        oh, ow = imgs.shape[-2:]
        result = [imgs]  

        for r in scales:
            nh, nw = max(1, int(r * oh)), max(1, int(r * ow))
            s = F.interpolate(imgs, size=(nh, nw), mode="bilinear", align_corners=False)

            if self.use_crop and nh > 1 and nw > 1:
                ch, cw = max(1, int(nh * random.uniform(*self.crop_ratio_range))), max(1, int(nw * random.uniform(*self.crop_ratio_range)))
                sh = random.randint(0, nh - ch) if nh > ch else 0
                sw = random.randint(0, nw - cw) if nw > cw else 0
                s = F.interpolate(s[:, :, sh:sh+ch, sw:sw+cw], size=(nh, nw), mode="bilinear", align_corners=False)

            s_global = F.interpolate(s, size=(oh, ow), mode="bilinear", align_corners=False)
            result.append(s_global)

            if mask is not None:
                m = mask[:, :1, :, :]  
                m_s = F.interpolate(m, size=(nh, nw), mode="bilinear", align_corners=False)
                m_r = F.interpolate(m_s, size=(oh, ow), mode="bilinear", align_corners=False)
                m_hard = (m_r > 0.5).float()
                local_fg = s_global * m_hard + imgs * (1 - m_hard)
                result.append(local_fg)

        return torch.cat(result)

    def _path_grad(self, model, adv, scales, txt_embeds, txt2img, proj, mask=None):
        B = adv.shape[0]
        total_grad = torch.zeros_like(adv)

        if scales:
            raw_weights = [float(s) for s in scales]
            path_weights = []
            for i in range(self.path_samples):
                if i < len(raw_weights):
                    path_weights.append(raw_weights[i])
                else:
                    path_weights.append(raw_weights[-1])
        else:
            path_weights = [1.0] * self.path_samples

        first_w = path_weights[0]
        weight_sum = sum(path_weights)

        for w in path_weights:
            sigma = self.path_sigma * w / first_w

            for sign in (1.0, -1.0):
                noise = torch.randn_like(adv) * sigma
                x = adv + sign * noise

                if scales:
                    emb = self.get_image_feat(model, self.get_scaled_imgs(x, scales, mask=mask))
                    nv = emb.shape[0] // B
                    loss = sum(
                        self.loss_func(emb[v*B:(v+1)*B], txt_embeds, txt2img, proj)
                        for v in range(nv)
                    ) / nv
                else:
                    loss = self.loss_func(self.get_image_feat(model, x), txt_embeds, txt2img, proj)

                gi = torch.autograd.grad(loss, adv)[0]
                total_grad += w * gi

        return total_grad / weight_sum

    def txt_guided_attack(self, model, imgs, txt2img, all_txt_supervisions, device,
                          scales=None, txt_embeds=None, return_grad=False, important_words=None):
        model.eval()
        dev = torch.device(device)
        imgs = imgs.to(dev)
        txt_embeds, all_txt_supervisions = txt_embeds.to(dev), all_txt_supervisions.to(dev)

        U, _, _ = torch.svd(all_txt_supervisions.T.to(torch.float32))
        k = min(U.size(1) - 1, 128)
        proj = (U[:, 1:k+1] @ U[:, 1:k+1].t()).to(dev) if k > 0 else torch.eye(U.size(0), device=dev)

        adv = torch.clamp(imgs + torch.empty_like(imgs).uniform_(-self.eps, self.eps), 0, 1)
        inner0, raw_attn = self.get_dino_masks(imgs, dev)
        r_in = inner0.clone()
        mom = torch.zeros_like(adv)
        last_adv = adv.clone()
        mm = 0.6

        for step in range(self.steps):
            if step > 0 and step % self.mask_update_interval == 0:
                n_in, raw_attn = self.get_dino_masks(adv.detach(), dev)
                r_in = mm * r_in + (1 - mm) * n_in

            adv = adv.detach().requires_grad_(True)
            grad = self._path_grad(model, adv, scales, txt_embeds, txt2img, proj, mask=r_in)
            grad = grad / (torch.norm(grad.view(grad.size(0), -1), p=1, dim=1).view(-1,1,1,1) + 1e-8)

            mom = self.momentum * mom + grad
            dir_ = mom.sign()

            inner_p, outer_p = self.step_size_inner * dir_, self.step_size_outer * dir_
            p = torch.where(inner0 > 0.5, inner_p, outer_p)
            last_adv = adv.detach().clone()

            adv = adv.detach() + p
            adv_in = torch.clamp(adv, imgs - self.eps_inner, imgs + self.eps_inner)
            adv_out = torch.clamp(adv, imgs - self.eps_outer, imgs + self.eps_outer)
            adv = torch.clamp(torch.where(inner0 > 0.5, adv_in, adv_out), 0, 1)

        if return_grad:
            return adv.detach(), last_adv, raw_attn.detach()
        return adv.detach(), last_adv

    def save_img(self, img_name, norm_img, save_dir="./mscoco_imgs"):
        import os
        os.makedirs(save_dir, exist_ok=True)
        norm_img = norm_img.detach().clamp(0.0, 1.0)
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        if pil_array.shape[0] == 1:
            pil_array = pil_array[0]
            pil_img = Image.fromarray(pil_array, mode="L")
        else:
            pil_array = np.transpose(pil_array, (1, 2, 0))
            pil_img = Image.fromarray(pil_array)
        pil_img.save(f"{save_dir}/{img_name}")

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)

class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2]):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios

    def loss_func(self, txt_embeds, img_embeds, label):
        return -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)

    def img_guided_attack(self, net, texts, img_embeds=None, beta_t=0.0):
        device = self.ref_net.device
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True,
                                     max_length=self.max_length, return_tensors='pt').to(device)
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        final_adverse = []
        for i, text in enumerate(texts):
            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)

            best_candidates = []
            for pos in range(len(words)):
                tgt_word = words[pos]
                if tgt_word in filter_words:
                    continue
                if keys[pos][0] > self.max_length - 2:
                    continue
                substitutes_ids = word_predictions[i, keys[pos][0]:keys[pos][1]]
                word_pred_scores = word_pred_scores_all[i, keys[pos][0]:keys[pos][1]]
                candidates = get_substitues(substitutes_ids, self.tokenizer, self.ref_net, 1,
                                            word_pred_scores, self.threshold_pred_score)
                valid_candidates = []
                for cand in candidates:
                    if cand == tgt_word or '##' in cand or cand in filter_words:
                        continue
                    valid_candidates.append(cand)
                if not valid_candidates:
                    continue

                current_text = ' '.join(final_words)
                replace_texts = [current_text]
                available_substitutes = [tgt_word]
                for cand in valid_candidates:
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[pos] = cand
                    replace_texts.append(' '.join(temp_replace))
                    available_substitutes.append(cand)

                replace_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                               max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                loss = self.loss_func(replace_embeds, img_embeds, i)
                candidate_losses = loss[1:]
                best_loss, best_idx = candidate_losses.max(dim=0)
                best_candidates.append((pos, available_substitutes[best_idx.item() + 1], best_loss.item()))

            best_candidates.sort(key=lambda x: x[2], reverse=True)
            selected = best_candidates[:self.num_perturbation]
            for pos, replacement_word, _ in selected:
                final_words[pos] = replacement_word

            final_adverse.append(' '.join(final_words))
        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()
    if sub_len == 0:
        return words
    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words

def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]
    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    all_substitutes = torch.tensor(all_substitutes)
    all_substitutes = all_substitutes[:24].to(device)
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1))
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words