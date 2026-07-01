"""Knowledge-distillation utilities for the FL+KD 3D point-cloud classification benchmark.

KD losses (11)
--------------
Vanilla (Hinton), Feature-L2, Attention transfer (Zagoruyko & Komodakis),
Multi-teacher, Self-distillation (Born-Again), Logit-MSE, Cosine logits, CRD
(contrastive, Tian et al.), DKD (Decoupled KD, Zhao et al. 2022), RKD
(Relational KD, Park et al. 2019), SP (Similarity-Preserving, Tung & Mori 2019).

Compact students: ``SmallPointNetCls``, ``SmallPointNet2ClsSsg`` and the
heterogeneous ``DGCNNClsStudent`` (DGCNN edge-conv backbone).

Convention note: the PointNet/PointNet++ ``get_model`` modules return
``log_softmax`` outputs rather than raw logits.  Every soft-target loss below is
written under that same convention so all methods receive the same inputs
(softmax is shift-invariant and log_softmax is idempotent on log-probs, so
softmax(log_softmax(z)/T) == softmax(z/T) exactly: the soft-target KL and the
hard-label CE terms follow the standard formulation; logit-MSE/cosine act on the
log-probability vectors, a consistent convention across the benchmark).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft-target losses
# ---------------------------------------------------------------------------

class VanillaDistillationLoss(nn.Module):
    """Classical Hinton-style KD: CE(student, y) + alpha * KL(T||S; T^2)."""

    def __init__(self, alpha=0.5, temperature=2.0, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.use_hard = use_hard

    def forward(self, student_logits, teacher_logits, target, trans_feat=None):
        if target is not None and target.dtype != torch.long:
            target = target.long()

        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        if not self.use_hard or target is None:
            return soft_loss

        hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), target)
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss


DistillationLoss = VanillaDistillationLoss


class FeatureDistillationLoss(nn.Module):
    """Hinton KD + L2 distance between normalised intermediate features."""

    def __init__(self, alpha=0.5, temperature=2.0, beta=0.5, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta
        self.use_hard = use_hard
        self.feature_adapter = None

    def extra_parameters(self):
        return list(self.feature_adapter.parameters()) if self.feature_adapter is not None else []

    def forward(self, student_logits, teacher_logits, target,
                student_features=None, teacher_features=None, trans_feat=None):
        if target is not None and target.dtype != torch.long:
            target = target.long()

        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            if student_features.dim() > 2:
                student_features = student_features.reshape(student_features.size(0), -1)
            if teacher_features.dim() > 2:
                teacher_features = teacher_features.reshape(teacher_features.size(0), -1)

            if student_features.size(1) != teacher_features.size(1):
                if self.feature_adapter is None:
                    self.feature_adapter = nn.Linear(student_features.size(1), teacher_features.size(1)).to(student_features.device)
                student_features = self.feature_adapter(student_features)

            student_features = F.normalize(student_features, p=2, dim=1)
            teacher_features = F.normalize(teacher_features, p=2, dim=1)
            feature_loss = F.mse_loss(student_features, teacher_features)

        if not self.use_hard or target is None:
            return self.alpha * soft_loss + self.beta * feature_loss

        hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), target)
        return (1 - self.alpha - self.beta) * hard_loss + self.alpha * soft_loss + self.beta * feature_loss


class AttentionDistillationLoss(nn.Module):
    """Attention-transfer KD (Zagoruyko & Komodakis 2017) on aggregated features."""

    def __init__(self, alpha=0.5, temperature=2.0, beta=0.5, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta
        self.use_hard = use_hard
        self.feature_adapter = None

    def extra_parameters(self):
        return list(self.feature_adapter.parameters()) if self.feature_adapter is not None else []

    @staticmethod
    def _attention_map(features):
        return torch.sum(torch.abs(features), dim=1, keepdim=True)

    def forward(self, student_logits, teacher_logits, target,
                student_features=None, teacher_features=None, trans_feat=None):
        if target is not None and target.dtype != torch.long:
            target = target.long()

        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        attention_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            if student_features.dim() > 2:
                student_features = student_features.reshape(student_features.size(0), -1)
            if teacher_features.dim() > 2:
                teacher_features = teacher_features.reshape(teacher_features.size(0), -1)

            if student_features.size(1) != teacher_features.size(1):
                if self.feature_adapter is None:
                    self.feature_adapter = nn.Linear(student_features.size(1), teacher_features.size(1)).to(student_features.device)
                student_features = self.feature_adapter(student_features)

            # Attention descriptor on the (global-pooled) PointNet++ feature:
            # per-sample squared activations L2-normalised across channels
            # (Zagoruyko & Komodakis style, adapted to a pooled feature).  After
            # the adapter both tensors share the teacher channel width.
            student_attention = F.normalize(student_features.pow(2), p=2, dim=1)
            teacher_attention = F.normalize(teacher_features.pow(2), p=2, dim=1)
            attention_loss = F.mse_loss(student_attention, teacher_attention)

        if not self.use_hard or target is None:
            return self.alpha * soft_loss + self.beta * attention_loss

        hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), target)
        return (1 - self.alpha - self.beta) * hard_loss + self.alpha * soft_loss + self.beta * attention_loss


class MultiTeacherDistillationLoss(nn.Module):
    """Ensemble KD: weighted average of teacher soft targets."""

    def __init__(self, alpha=0.5, temperature=2.0, teacher_weights=None, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_weights = teacher_weights
        self.use_hard = use_hard

    def forward(self, student_logits, teacher_logits_list, target, trans_feat=None):
        if target is not None and target.dtype != torch.long:
            target = target.long()

        n = len(teacher_logits_list)
        if self.teacher_weights is None:
            self.teacher_weights = [1.0 / n] * n

        soft_loss = 0.0
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        for w, t_logits in zip(self.teacher_weights, teacher_logits_list):
            soft_teacher = F.softmax(t_logits / self.temperature, dim=1)
            soft_loss = soft_loss + w * F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        if not self.use_hard or target is None:
            return soft_loss
        hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), target)
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss


class SelfDistillationLoss(VanillaDistillationLoss):
    """Born-Again-Networks-style self-distillation; numerically identical to Vanilla KD."""


class LogitMSEDistillationLoss(nn.Module):
    """MSE on raw logits + CE on hard labels."""

    def __init__(self, alpha=0.5, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.use_hard = use_hard

    def forward(self, student_logits, teacher_logits, target, trans_feat=None):
        mse_loss = F.mse_loss(student_logits, teacher_logits)
        if not self.use_hard or target is None:
            return mse_loss
        if target.dtype != torch.long:
            target = target.long()
        hard_loss = F.cross_entropy(student_logits, target)
        return (1 - self.alpha) * hard_loss + self.alpha * mse_loss


class CosineDistillationLoss(nn.Module):
    """1 - cos(student, teacher) on l2-normalised logits + CE on hard labels."""

    def __init__(self, alpha=0.5, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.use_hard = use_hard

    def forward(self, student_logits, teacher_logits, target, trans_feat=None):
        s = F.normalize(student_logits, p=2, dim=1)
        t = F.normalize(teacher_logits, p=2, dim=1)
        cos_loss = 1 - torch.mean(torch.sum(s * t, dim=1))
        if not self.use_hard or target is None:
            return cos_loss
        if target.dtype != torch.long:
            target = target.long()
        hard_loss = F.cross_entropy(student_logits, target)
        return (1 - self.alpha) * hard_loss + self.alpha * cos_loss


class CRDDistillationLoss(nn.Module):
    """Contrastive Representation Distillation (Tian, Krishnan, Isola 2020).

    A simplified in-batch ("memory-free") NT-Xent variant: the positive pair for
    each sample is the teacher embedding of the same sample; negatives are the
    teacher embeddings of every other sample in the batch.  Student/teacher
    features are projected through learnable ``feat_dim`` heads.
    """

    def __init__(self, alpha=0.5, temperature=0.07, feat_dim=128, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feat_dim = feat_dim
        self.use_hard = use_hard
        self.student_head = None
        self.teacher_head = None

    def extra_parameters(self):
        params = []
        if self.student_head is not None:
            params += list(self.student_head.parameters())
        if self.teacher_head is not None:
            params += list(self.teacher_head.parameters())
        return params

    def _maybe_init_heads(self, s_dim, t_dim, device):
        if self.student_head is None:
            self.student_head = nn.Linear(s_dim, self.feat_dim).to(device)
        if self.teacher_head is None:
            self.teacher_head = nn.Linear(t_dim, self.feat_dim).to(device)

    def forward(self, student_logits, teacher_logits, target,
                student_features=None, teacher_features=None, trans_feat=None):
        if target is not None and target.dtype != torch.long:
            target = target.long()

        if student_features is None or teacher_features is None:
            return LogitMSEDistillationLoss(alpha=self.alpha, use_hard=self.use_hard)(
                student_logits, teacher_logits, target)

        if student_features.dim() > 2:
            student_features = student_features.reshape(student_features.size(0), -1)
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.reshape(teacher_features.size(0), -1)

        self._maybe_init_heads(student_features.size(1), teacher_features.size(1), student_features.device)

        s = F.normalize(self.student_head(student_features), p=2, dim=1)
        t = F.normalize(self.teacher_head(teacher_features), p=2, dim=1)

        logits = torch.matmul(s, t.t()) / max(1e-6, self.temperature)
        labels = torch.arange(s.size(0), device=s.device)
        contrastive = F.cross_entropy(logits, labels)

        if not self.use_hard or target is None:
            return contrastive

        hard_loss = F.cross_entropy(student_logits, target)
        return (1 - self.alpha) * hard_loss + self.alpha * contrastive


class DKDDistillationLoss(nn.Module):
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022).

    Splits the KD term into Target-Class KD (TCKD, a binary target-vs-rest
    distribution) and Non-target-Class KD (NCKD, over the non-target classes
    re-normalised).  ``alpha`` weights TCKD and ``beta`` weights NCKD (defaults
    1 and 8).  Requires hard targets; in the unlabeled ablation it
    falls back to vanilla KL so the run still completes.
    """

    def __init__(self, alpha=1.0, beta=8.0, temperature=4.0, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.use_hard = use_hard

    def forward(self, student_logits, teacher_logits, target, trans_feat=None):
        T = self.temperature
        if target is None:
            soft_student = F.log_softmax(student_logits / T, dim=1)
            soft_teacher = F.softmax(teacher_logits / T, dim=1)
            return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)

        target = target.long()
        num_classes = student_logits.size(1)
        gt_mask = F.one_hot(target, num_classes).float()
        other_mask = 1.0 - gt_mask

        p_s = F.softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        ps_bin = torch.stack([(p_s * gt_mask).sum(1), (p_s * other_mask).sum(1)], dim=1).clamp_min(1e-6)
        pt_bin = torch.stack([(p_t * gt_mask).sum(1), (p_t * other_mask).sum(1)], dim=1).clamp_min(1e-6)
        tckd = F.kl_div(ps_bin.log(), pt_bin, reduction='batchmean') * (T * T)

        s_nt = F.log_softmax(student_logits / T - 1000.0 * gt_mask, dim=1)
        t_nt = F.softmax(teacher_logits / T - 1000.0 * gt_mask, dim=1)
        nckd = F.kl_div(s_nt, t_nt, reduction='batchmean') * (T * T)

        dkd = self.alpha * tckd + self.beta * nckd
        if not self.use_hard:
            return dkd
        hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), target)
        return hard_loss + dkd


class RKDDistillationLoss(nn.Module):
    """Relational Knowledge Distillation (Park et al., CVPR 2019).

    Matches pairwise distances (RKD-D) and triplet angles (RKD-A) of the student
    and teacher feature embeddings.  Combined with the soft-target term under the
    same (alpha soft, beta relational) template as the other feature losses.
    """

    def __init__(self, alpha=0.5, beta=0.5, temperature=2.0, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.use_hard = use_hard

    @staticmethod
    def _pdist(e):
        # Squared pairwise distances, then sqrt with an epsilon floor.  Clamping
        # to a small positive value (not 0) is essential: d/dx sqrt(x) is infinite
        # at x=0, so the zero diagonal would otherwise produce NaN gradients.
        e_sq = (e ** 2).sum(dim=1)
        sq = e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2 * (e @ e.t())
        return sq.clamp_min(1e-12).sqrt()

    def _rkd_distance(self, s, t):
        with torch.no_grad():
            td = self._pdist(t)
            pos = td > 0
            mean_td = td[pos].mean() if pos.any() else torch.tensor(1.0, device=t.device)
            td = td / mean_td
        sd = self._pdist(s)
        pos = sd > 0
        mean_sd = sd[pos].mean() if pos.any() else torch.tensor(1.0, device=s.device)
        sd = sd / mean_sd
        return F.smooth_l1_loss(sd, td)

    def _rkd_angle(self, s, t):
        with torch.no_grad():
            td = F.normalize(t.unsqueeze(0) - t.unsqueeze(1), p=2, dim=2)
            t_angle = torch.bmm(td, td.transpose(1, 2)).view(-1)
        sd = F.normalize(s.unsqueeze(0) - s.unsqueeze(1), p=2, dim=2)
        s_angle = torch.bmm(sd, sd.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(s_angle, t_angle)

    def forward(self, student_logits, teacher_logits, target,
                student_features=None, teacher_features=None, trans_feat=None):
        if target is not None and target.dtype != torch.long:
            target = target.long()
        T = self.temperature
        soft_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                             F.softmax(teacher_logits / T, dim=1),
                             reduction='batchmean') * (T * T)

        rel = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            s = student_features.reshape(student_features.size(0), -1)
            t = teacher_features.reshape(teacher_features.size(0), -1)
            rel = self._rkd_distance(s, t) + self._rkd_angle(s, t)

        if not self.use_hard or target is None:
            return self.alpha * soft_loss + self.beta * rel
        hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), target)
        return (1 - self.alpha - self.beta) * hard_loss + self.alpha * soft_loss + self.beta * rel


class SPDistillationLoss(nn.Module):
    """Similarity-Preserving KD (Tung & Mori, ICCV 2019).

    Encourages the student to preserve the teacher's pairwise sample-similarity
    structure: G = normalise(F F^T); loss = ||G_s - G_t||^2.  Combined with the
    soft-target term under the same (alpha soft, beta similarity) template.
    """

    def __init__(self, alpha=0.5, beta=0.5, temperature=2.0, use_hard=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.use_hard = use_hard

    @staticmethod
    def _gram(f):
        g = f @ f.t()
        return F.normalize(g, p=2, dim=1)

    def forward(self, student_logits, teacher_logits, target,
                student_features=None, teacher_features=None, trans_feat=None):
        if target is not None and target.dtype != torch.long:
            target = target.long()
        T = self.temperature
        soft_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                             F.softmax(teacher_logits / T, dim=1),
                             reduction='batchmean') * (T * T)

        sp = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            s = student_features.reshape(student_features.size(0), -1)
            t = teacher_features.reshape(teacher_features.size(0), -1)
            sp = (self._gram(s) - self._gram(t)).pow(2).mean()

        if not self.use_hard or target is None:
            return self.alpha * soft_loss + self.beta * sp
        hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), target)
        return (1 - self.alpha - self.beta) * hard_loss + self.alpha * soft_loss + self.beta * sp


# ---------------------------------------------------------------------------
# Compact student backbones (same family as the teacher)
# ---------------------------------------------------------------------------

class SmallPointNetCls(nn.Module):
    """Lightweight PointNet (no T-Net) classifier."""

    def __init__(self, num_class, normal_channel=True):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.conv1 = nn.Conv1d(in_channel, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_class)

        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        if self.normal_channel:
            norm = x[:, 3:, :]
            x = x[:, :3, :]
            x = torch.cat([x, norm], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        feat = F.relu(self.bn4(self.conv4(x)))

        pooled = torch.max(feat, 2)[0]
        x = F.relu(self.bn5(self.fc1(pooled)))
        x = F.relu(self.bn6(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # Match the SmallPointNet2 / DGCNN students and the PointNet++ teacher,
        # which all return log_softmax.  Without this, logit-space losses
        # (logit_mse / cosine) would compare this student's RAW logits against the
        # teacher's log-probabilities.  Softmax/log_softmax based losses are
        # unaffected (log_softmax is idempotent; softmax(log_softmax(z)/T)==softmax(z/T)).
        x = F.log_softmax(x, dim=-1)
        # Return the pooled penultimate feature so feature-based KD has a signal.
        return x, pooled.view(pooled.size(0), -1, 1)


class SmallPointNet2ClsSsg(nn.Module):
    """Slim PointNet++ SSG used as the same-family (compressed) student."""

    def __init__(self, num_class, normal_channel=True):
        super().__init__()
        import sys
        import os
        import importlib

        models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        sys.path.append(models_path)
        try:
            from pointnet2_utils import PointNetSetAbstraction
        except ImportError:
            sys.path.insert(0, models_path)
            try:
                from pointnet2_utils import PointNetSetAbstraction
            except ImportError:
                pointnet2_utils = importlib.import_module('pointnet2_utils')
                PointNetSetAbstraction = pointnet2_utils.PointNetSetAbstraction

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=16, in_channel=in_channel,
                                          mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=64 + 3,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3,
                                          mlp=[128, 256, 512], group_all=True)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B = xyz.size(0)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 512)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


# ---------------------------------------------------------------------------
# Heterogeneous student: DGCNN (Wang et al., 2019)
# ---------------------------------------------------------------------------

def _knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(2, 1)
    return pairwise.topk(k=k, dim=-1)[1]


def _get_graph_feature(x, k=20):
    B, C, N = x.size()
    idx = _knn(x, k=k)
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)

    x_flat = x.transpose(2, 1).contiguous().view(B * N, -1)
    neigh = x_flat[idx].view(B, N, k, C)
    expanded = x.transpose(2, 1).unsqueeze(2).expand(-1, -1, k, -1)
    feature = torch.cat((neigh - expanded, expanded), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNNClsStudent(nn.Module):
    """A compact DGCNN classifier - heterogeneous-architecture student."""

    def __init__(self, num_class, normal_channel=False, k=20, emb_dims=256, dropout=0.4):
        super().__init__()
        self.k = k
        self.normal_channel = normal_channel
        in_channel = 6 if normal_channel else 3

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel * 2, 32, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(32 + 64 + 128, emb_dims, 1, bias=False),
                                   nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2))

        self.fc1 = nn.Linear(emb_dims * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        if self.normal_channel and x.size(1) == 6:
            xyz = x[:, :3, :]
            norm = x[:, 3:, :]
            x = torch.cat([xyz, norm], dim=1)

        f1 = _get_graph_feature(x, k=self.k)
        f1 = self.conv1(f1).max(dim=-1, keepdim=False)[0]
        f2 = _get_graph_feature(f1, k=self.k)
        f2 = self.conv2(f2).max(dim=-1, keepdim=False)[0]
        f3 = _get_graph_feature(f2, k=self.k)
        f3 = self.conv3(f3).max(dim=-1, keepdim=False)[0]
        cat = torch.cat([f1, f2, f3], dim=1)
        feat = self.conv4(cat)

        pooled_max = feat.max(dim=-1)[0]
        pooled_avg = feat.mean(dim=-1)
        pooled = torch.cat([pooled_max, pooled_avg], dim=1)

        x = F.leaky_relu(self.bn1(self.fc1(pooled)), 0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.dp2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x, feat.max(dim=-1, keepdim=True)[0]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_inference_time(model, test_loader, device, num_runs=100):
    model.eval()
    timings = []

    points = None
    for points, _ in test_loader:
        points = points.transpose(2, 1).to(device)
        break
    if points is None:
        return float('nan')

    with torch.no_grad():
        for _ in range(10):
            _ = model(points)

    with torch.no_grad():
        if device.type == 'cuda':
            try:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                for _ in range(num_runs):
                    starter.record()
                    _ = model(points)
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender))
            except Exception as exc:  # pragma: no cover
                print(f'CUDA timing fell back to CPU: {exc}')
                import time as _time
                for _ in range(num_runs):
                    t0 = _time.time()
                    _ = model(points)
                    timings.append((_time.time() - t0) * 1000.0)
        else:
            import time as _time
            for _ in range(num_runs):
                t0 = _time.time()
                _ = model(points)
                timings.append((_time.time() - t0) * 1000.0)

    return float(sum(timings) / max(1, len(timings)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gflops_per_forward(model, sample_input):
    """Rough GFLOPs/forward estimate (best-effort); NaN if profiling fails."""
    try:
        from torch.profiler import profile, ProfilerActivity  # type: ignore
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
            with torch.no_grad():
                model(sample_input)
        total_flops = sum(evt.flops or 0 for evt in prof.events())
        return total_flops / 1e9 if total_flops > 0 else float('nan')
    except Exception:
        return float('nan')
