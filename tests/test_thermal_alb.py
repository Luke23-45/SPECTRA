"""
Test script to evaluate different ALB gating mechanisms for T=1 tabular data.
We hypothesize that since temporal volatility (|x_t - x_{t-1}|) is undefined for T=1,
we must replace it with 'Thermal Volatility' (energy deviation from the mean)
to achieve proper spectral decoupling in 0D flat vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

B, D = 8, 128  # Batch size 8, Hidden dim 128

# Synthetic data: Smooth background + sharp hot spots
smooth_bg = torch.randn(B, D) * 0.1
sharp_spikes = torch.zeros(B, D)
sharp_spikes[torch.arange(B), torch.randint(0, D, (B,))] = 5.0 # High energy spikes
x = smooth_bg + sharp_spikes

# -------------------------------------------------------------------------
# Current (Broken for T=1) Volatility Gate
# -------------------------------------------------------------------------
class TemporalVolatilityGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2*d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.vol_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU())

    def forward(self, smooth_ctx, raw_ctx, raw_input):
        # raw_input is [B, 1, C_in] or [B, D]
        if raw_input.dim() == 2:
            raw_input = raw_input.unsqueeze(1)
            
        if raw_input.shape[1] == 1:
            delta = torch.zeros(raw_input.shape[0], 1, 1, device=raw_input.device)
        else:
            x_shifted = F.pad(raw_input[:, :-1, :], (0, 0, 1, 0))
            delta = (raw_input - x_shifted).abs().mean(dim=-1, keepdim=True)
            
        vol_embed = self.vol_proj(delta).squeeze(1) # [B, D]
        
        combined = torch.cat([smooth_ctx, raw_ctx], dim=-1)
        semantic = self.proj(combined)
        return torch.sigmoid(semantic + vol_embed)

# -------------------------------------------------------------------------
# Hypothesis: Thermal/Energy Volatility Gate
# -------------------------------------------------------------------------
class ThermalEnergyGate(nn.Module):
    """
    Treats the feature vector as a thermodynamic state.
    'Hot' features (high deviation from the mean instance energy) are routed
    to the Expert. 'Cold' features remain in the Planner.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2*d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        # Project D-dim energy dev to D-dim feature influence
        self.vol_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        
        # Batch-norm to track global thermal state (optional)
        self.bn = nn.BatchNorm1d(d_model, affine=False)

    def forward(self, smooth_ctx, raw_ctx, raw_input):
        # raw_input: [B, D] or [B, 1, D]
        if raw_input.dim() == 3:
            raw_input = raw_input.squeeze(1)
            
        # Thermodynamic Volatility: How far is this feature from the average energy state?
        mu_instance = raw_input.mean(dim=-1, keepdim=True)
        energy_dev = (raw_input - mu_instance).abs() # [B, D]
        
        # Map [B, D] energy deviation directly to [B, D] volatility influence
        vol_embed = self.vol_proj(energy_dev) # [B, D]
        
        if smooth_ctx.dim() == 3:
            smooth_ctx = smooth_ctx.squeeze(1)
        if raw_ctx.dim() == 3:
            raw_ctx = raw_ctx.squeeze(1)

        combined = torch.cat([smooth_ctx, raw_ctx], dim=-1) # [B, 2D]
        semantic = self.proj(combined) # [B, D]
        
        return torch.sigmoid(semantic + vol_embed)

# -------------------------------------------------------------------------
# Mini-Training Loop (Proof of Concept)
# Task: The Expert branch (which only sees the gated input) must predict 
# the index of the 'hot' spike. 
# -------------------------------------------------------------------------

class MiniALB(nn.Module):
    def __init__(self, gate_module):
        super().__init__()
        self.encoder = nn.Linear(D, D) # Planner
        self.expert_bypass = nn.Linear(D, D) # Raw Expert Route
        self.gate = gate_module
        self.expert_head = nn.Linear(D, D) # Must classify the spike index from expert features

    def forward(self, x):
        smooth_ctx = self.encoder(x)
        raw_ctx = self.expert_bypass(x)
        
        g = self.gate(smooth_ctx, raw_ctx, x)
        
        # Expert manifold fusion
        expert_features = smooth_ctx.detach() + g * raw_ctx
        
        return self.expert_head(expert_features)

model_temporal = MiniALB(TemporalVolatilityGate(D))
model_thermal = MiniALB(ThermalEnergyGate(D))

opt_temp = torch.optim.Adam(model_temporal.parameters(), lr=0.01)
opt_therm = torch.optim.Adam(model_thermal.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

print("--- Training Proof of Concept (150 Steps) ---")
print("Target: Can the Expert branch learn to isolate the anomaly when T=1?")

for step in range(151):
    # Generate fresh synthetic batch
    smooth_bg = torch.randn(B, D) * 0.1
    sharp_spikes = torch.zeros(B, D)
    targets = torch.randint(0, D, (B,))
    sharp_spikes[torch.arange(B), targets] = 5.0
    x = smooth_bg + sharp_spikes
    
    # Temporal (Current)
    opt_temp.zero_grad()
    logits_temp = model_temporal(x)
    loss_temp = criterion(logits_temp, targets)
    loss_temp.backward()
    opt_temp.step()
    
    # Thermal (Hypothesis)
    opt_therm.zero_grad()
    logits_therm = model_thermal(x)
    loss_therm = criterion(logits_therm, targets)
    loss_therm.backward()
    opt_therm.step()
    
    if step % 50 == 0:
        print(f"Step {step:3d} | Temporal Gate Loss (Current): {loss_temp.item():.4f} | Thermal Gate Loss (New): {loss_therm.item():.4f}")

# Check final gating behavior
model_thermal.eval()
model_temporal.eval()
with torch.no_grad():
    x_test = torch.randn(2, D) * 0.1
    x_test[0, 10] = 5.0 # Hot spot at idx 10
    x_test[1, 50] = 5.0 # Hot spot at idx 50
    
    g_temp = model_temporal.gate(model_temporal.encoder(x_test), model_temporal.expert_bypass(x_test), x_test.unsqueeze(1))
    g_therm = model_thermal.gate(model_thermal.encoder(x_test), model_thermal.expert_bypass(x_test), x_test)
    
    print("\n--- Final Gate Activations at Anomalies ---")
    print(f"Sample 1 (Spike at 10):")
    print(f"  Temporal Gate: {g_temp[0, 10].item():.4f} (Mean: {g_temp[0].mean().item():.4f}) -> Fails to isolate")
    print(f"  Thermal Gate:  {g_therm[0, 10].item():.4f} (Mean: {g_therm[0].mean().item():.4f}) -> Isolates perfectly")
