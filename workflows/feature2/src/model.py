import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FiLM(nn.Module):
    def __init__(self, feat_dim, cond_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim * 2),
        )

    def forward(self, x, cond):
        """
        x: B,C,H,W
        cond: B,cond_dim
        """
        B, C, H, W = x.shape
        params = self.fc(cond)  # B, 2C
        gamma, beta = params[:, :C], params[:, C:]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return x * (1 + gamma) + beta


class TinyRelightNet(nn.Module):
    def __init__(self, in_ch=10, base_ch=32):
        """
        in_ch = input_rgb(3) + physics(3) + normals(3) + depth(1) = 10
        """
        super().__init__()
        # -------- Encoder --------
        self.enc1 = ConvBlock(in_ch, base_ch)               # 10 -> 32, H
        self.enc2 = ConvBlock(base_ch, base_ch * 2)         # 32 -> 64, H/2
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)     # 64 -> 128, H/4

        self.pool = nn.AvgPool2d(2)

        # Bottleneck (H/8)
        self.bott = ConvBlock(base_ch * 4, base_ch * 4)     # 128 -> 128

        # FiLM (cond_dim=4: lx, ly, lz, intensity)
        self.film_bott = FiLM(base_ch * 4, cond_dim=4)      # 128
        self.film_enc2 = FiLM(base_ch * 2, cond_dim=4)      # 64

        # -------- Decoder --------
        # H/8 -> H/4
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)   # 128 -> 64
        # d3 (64) + e3 (128) = 192 = base_ch*6
        self.dec3 = ConvBlock(base_ch * 6, base_ch * 2)                  # 192 -> 64

        # H/4 -> H/2
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)       # 64 -> 32
        # d2 (32) + e2 (64) = 96 = base_ch*3
        self.dec2 = ConvBlock(base_ch * 3, base_ch)                      # 96 -> 32

        # H/2 -> H
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, 2)           # 32 -> 32
        # up1 (32) + e1 (32) = 64 = base_ch*2
        self.dec1 = ConvBlock(base_ch * 2, base_ch)                      # 64 -> 32

        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)              # 32 -> 3

    def forward(self, x, light_cond):
        """
        x: B,in_ch,H,W
        light_cond: B,4  (lx, ly, lz, intensity)
        """
        # ---- Encoder ----
        e1 = self.enc1(x)              # B,32,H,    W
        e2 = self.enc2(self.pool(e1))  # B,64,H/2,  W/2
        e3 = self.enc3(self.pool(e2))  # B,128,H/4, W/4
        b = self.bott(self.pool(e3))   # B,128,H/8, W/8

        # ---- FiLM conditioning ----
        b = self.film_bott(b, light_cond)    # B,128,H/8,W/8
        e2 = self.film_enc2(e2, light_cond)  # B,64,H/2,W/2

        # ---- Decoder ----
        # H/8 -> H/4
        d3 = self.up3(b)                                  # B,64,H/4,W/4
        d3 = self.dec3(torch.cat([d3, e3], dim=1))        # B,64,H/4,W/4

        # H/4 -> H/2
        d2 = self.up2(d3)                                 # B,32,H/2,W/2
        d2 = self.dec2(torch.cat([d2, e2], dim=1))        # B,32,H/2,W/2

        # H/2 -> H
        u1 = self.up1(d2)                                 # B,32,H,W
        u1 = self.dec1(torch.cat([u1, e1], dim=1))        # B,32,H,W

        out = self.out_conv(u1)        # B,3,H,W
        out = torch.sigmoid(out)       # [0,1]
        return out


def load_model(weights_path: str, device: str = 'cpu'):
    """Instantiate TinyRelightNet model and load weights from a checkpoint (.pth).

    The loader tries a few fallback patterns so it will work with common checkpoints:
    - full state_dict (keys directly match model.state_dict())
    - checkpoint['state_dict']
    - checkpoint['model_state_dict']
    - checkpoint['model']
    """
    model = TinyRelightNet(in_ch=10, base_ch=32)
    map_location = torch.device(device)
    ckpt = torch.load(weights_path, map_location=map_location, weights_only=False)

    # try common checkpoint formats
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            state = ckpt['model_state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
        else:
            # assume ckpt is a raw state_dict or contains layer keys
            state = ckpt
    else:
        state = ckpt

    # Remove possible 'module.' prefixes produced by DataParallel training
    new_state = {}
    for k, v in state.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_k] = v

    # attempt to load
    msg = model.load_state_dict(new_state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"Loaded weights from {weights_path}. Missing keys: {len(msg.missing_keys)} Unexpected keys: {len(msg.unexpected_keys)}")
    else:
        print(f"Loaded weights from {weights_path} successfully.")
    model.to(map_location)
    model.eval()
    return model
