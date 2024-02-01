from torch import nn, Tensor
from scaling import ScaledConv1d, ScaledLinear
import argparse 
from icefall.utils import make_pad_mask

def add_omega_arguments(parser : argparse.ArgumentParser):
    group = parser.add_argument_group(title = "Weight prediction submodule (Omega) related options")
    
    group.add_argument(
        "--alpha-actv",
        type=str,
        default="sigmoid",
    )
    
    group.add_argument(
        "--omega-type",
        type=str,
        default="Mean",
        help=(
            "Check codes to find out what omegas are supported. "
            "'-' demarcates the hidden channel dimension. Not necessary depending on omega type. "
            "Use 0 to mean encoder_dim. "
            "',' demarcates fields for omega."
        )
    )
    
    group.add_argument(
        "--omega-init-scale",
        type=str,
        default="D ** -0.5",
        help="All 'D' will be converted to `params.encoder_dim`."
    )

def get_omega_model(params) -> "Omega":
    *hid_channel, omega_type = params.omega_type.split("-")    
    
    if hid_channel and hid_channel[0] == "D":
        hid_channel = params.encoder_out_dim
    elif hid_channel:
        hid_channel = int(hid_channel[0])
    else:
        hid_channel = None

    actv_f = _parse_actv_f(params.alpha_actv, hid_channels=hid_channel)
    
    omega_type, *args = omega_type.split(",")
    omega_init_scale = params.omega_init_scale.replace("D", "params.encoder_out_dim")
    omega_init_scale = eval(omega_init_scale)
    
    if "CnnFc" in omega_type:
        assert hid_channel is not None, hid_channel
        omega = CnnFcOmega(
            in_channels=params.encoder_out_dim,
            hid_channels=hid_channel,
            kernel_size=int(args[0]) if args else 3 ,
            actv_f=actv_f,
            initial_scale=omega_init_scale,
        )
    elif "Mean" in omega_type:
        assert hid_channel is None, hid_channel
        omega = MeanOmega(actv_f=actv_f)
    else:
        raise NotImplementedError(f"Omega type {omega_type} not implemented.")
    
    return omega

def _parse_actv_f(raw_actv_f : str, hid_channels = None):
    
    if "sigmoid" in raw_actv_f:
        assert raw_actv_f == "sigmoid", raw_actv_f
        actv_f = nn.Sigmoid()
    elif "abs" in raw_actv_f:
        assert raw_actv_f == "abs", raw_actv_f
        class Absolute(nn.Module):
            def __init__(self):
                super(Absolute, self).__init__()
            def forward(self, x: Tensor) -> Tensor:
                return x.abs()
        actv_f = Absolute()
    else:
        raise NotImplementedError(f"{raw_actv_f} not implemented.")

    return actv_f

class Omega(nn.Module):
    """
    Family of modules that produces weights per acoustic frame
    """
    def __init__(self):
        super(Omega, self).__init__()
    
    def forward(self, x: Tensor, src_lens : Tensor) -> Tensor:
        # x = (B, T, C)
        raise NotImplementedError("Plaese implement in a subclass")

class MeanOmega(Omega):
    def __init__(self, actv_f : nn.Module):
        super().__init__()
        self.actv_f = actv_f
    
    def forward(self, x: Tensor, src_lens : Tensor) -> Tensor:
        alpha_mask = make_pad_mask(src_lens)[..., None]
        x = self.actv_f(x.contiguous().mean(dim=-1, keepdim=True))
        return x.masked_fill(alpha_mask, 0.)

class CnnFcOmega(Omega):
    def __init__(
        self, 
        in_channels : int, 
        hid_channels : int, 
        kernel_size : int, 
        actv_f : nn.Module, 
        initial_scale : float, 
        dropout = 0.1,
    ):
        super().__init__()
        self.conv1d = ScaledConv1d(
            in_channels=in_channels,
            out_channels=hid_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            bias=False
        )
        self.linear = ScaledLinear(
            in_features=hid_channels,
            out_features=1,
            bias=False,
            initial_scale=initial_scale,
        )
        self.actv_f = actv_f
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: Tensor, src_lens: Tensor) -> Tensor:
        alpha_mask = make_pad_mask(src_lens)[..., None]
        x = x.permute(0, 2, 1).contiguous() # (B, T, C) -> (B, C, T)
        x = self.conv1d(x).permute(0, 2, 1).contiguous() # (B, C, T) -> (B, T, C)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.actv_f(x)
        return x.masked_fill(alpha_mask, 0.)