import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    _has_vit = True
except Exception:
    _has_vit = False

from mrtf.preprocessing import preprocess_voice, preprocess_mri, preprocess_sensor


class VoiceCNNBiLSTM(nn.Module):
    def __init__(self, d=512, voice_freq_bins=128):
        super().__init__()
        self.voice_freq_bins = voice_freq_bins
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.proj = nn.Linear(128 * (voice_freq_bins // 4), 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.out = nn.Linear(512, d)

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.proj(x)
        out, _ = self.lstm(x)
        return self.out(out[:, -1, :])


class MRIEncoderViT(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        if _has_vit:
            weights = ViT_B_16_Weights.DEFAULT
            self.vit = vit_b_16(weights=weights)
            in_features = self.vit.heads.head.in_features
            self.vit.heads = nn.Identity()
            self.proj = nn.Linear(in_features, d)
        else:
            self.vit = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.proj = nn.Linear(128, d)

    def forward(self, x):
        if _has_vit:
            return self.proj(self.vit(x))
        return self.proj(self.vit(x).flatten(1))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.conv(x)[..., : x.shape[-1]]
        y = F.relu(y)
        y = self.dropout(y)
        return y + self.res(x)


class SensorTemporalEncoder(nn.Module):
    def __init__(self, d=512, sensor_len=200, sensor_channels=6, n_heads=8, n_layers=2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(sensor_channels, 64, dilation=1),
            TCNBlock(64, 128, dilation=2),
            TCNBlock(128, 256, dilation=4),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, sensor_len, 256) * 0.01)
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=n_heads, dim_feedforward=512, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(256, d)

    def forward(self, x):
        z = self.tcn(x).permute(0, 2, 1)
        z = z + self.pos_emb[:, : z.shape[1], :]
        z = self.tr(z).permute(0, 2, 1)
        pooled = self.pool(z).squeeze(-1)
        return self.out(pooled)


class CrossAttention(nn.Module):
    def __init__(self, d=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, q, k, v):
        out, w = self.mha(q, k, v, need_weights=True, average_attn_weights=True)
        x = self.ln(q + out)
        x = self.ln2(x + self.ff(x))
        return x, w


class CAFT(nn.Module):
    def __init__(self, d=512, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.cross_vm = CrossAttention(d, n_heads, dropout)
        self.cross_vs = CrossAttention(d, n_heads, dropout)
        self.cross_mv = CrossAttention(d, n_heads, dropout)
        self.cross_ms = CrossAttention(d, n_heads, dropout)
        self.cross_sv = CrossAttention(d, n_heads, dropout)
        self.cross_sm = CrossAttention(d, n_heads, dropout)

        self.omega = nn.Parameter(torch.ones(6))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ln = nn.LayerNorm(d)

    def forward(self, Ev, Em, Es):
        Ev1, Em1, Es1 = Ev.unsqueeze(1), Em.unsqueeze(1), Es.unsqueeze(1)

        v_m, w_vm = self.cross_vm(Ev1, Em1, Em1)
        v_s, w_vs = self.cross_vs(Ev1, Es1, Es1)
        m_v, w_mv = self.cross_mv(Em1, Ev1, Ev1)
        m_s, w_ms = self.cross_ms(Em1, Es1, Es1)
        s_v, w_sv = self.cross_sv(Es1, Ev1, Ev1)
        s_m, w_sm = self.cross_sm(Es1, Em1, Em1)

        ome = F.softmax(self.omega, dim=0)

        Ev_hat = ome[0] * v_m + ome[1] * v_s
        Em_hat = ome[2] * m_v + ome[3] * m_s
        Es_hat = ome[4] * s_v + ome[5] * s_m

        tokens = torch.cat([Ev_hat, Em_hat, Es_hat], dim=1)
        H = self.ln(self.tr(tokens))
        fused = H.mean(dim=1)

        attn_info = {"omega": ome.detach(), "w_vm": w_vm.detach(), "w_vs": w_vs.detach(),
                     "w_mv": w_mv.detach(), "w_ms": w_ms.detach(), "w_sv": w_sv.detach(), "w_sm": w_sm.detach()}
        return fused, attn_info


class MRTFClassifierHead(nn.Module):
    def __init__(self, d=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d // 2, 1))

    def forward(self, H):
        return torch.sigmoid(self.net(H)).squeeze(-1)


def fusion_regularization(Ev, Em, Es):
    return (Ev - Em).pow(2).mean() + (Ev - Es).pow(2).mean() + (Em - Es).pow(2).mean()


class MRTFModel(nn.Module):
    def __init__(self, cfg, rasl_module):
        super().__init__()
        self.cfg = cfg
        self.voice_enc = VoiceCNNBiLSTM(d=cfg.d, voice_freq_bins=cfg.voice_freq_bins)
        self.mri_enc = MRIEncoderViT(d=cfg.d)
        self.sensor_enc = SensorTemporalEncoder(d=cfg.d, sensor_len=cfg.sensor_len, sensor_channels=cfg.sensor_channels)
        self.caft = CAFT(d=cfg.d)
        self.clf = MRTFClassifierHead(d=cfg.d, dropout=0.3)
        self.rasl = rasl_module

        self.ln_v = nn.LayerNorm(cfg.d)
        self.ln_m = nn.LayerNorm(cfg.d)
        self.ln_s = nn.LayerNorm(cfg.d)

    def forward(self, voice, mri, sensor):
        voice = preprocess_voice(voice)
        mri = preprocess_mri(mri)
        sensor = preprocess_sensor(sensor)

        Ev = self.ln_v(self.voice_enc(voice))
        Em = self.ln_m(self.mri_enc(mri))
        Es = self.ln_s(self.sensor_enc(sensor))

        H, attn = self.caft(Ev, Em, Es)
        yhat = self.clf(H)
        return yhat, (Ev, Em, Es, H, attn)
