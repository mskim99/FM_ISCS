import torch
from torch.nn import functional as F


class ZAxisSuperResolution:
    def __init__(self, factor: int):
        self.factor = factor
        self.M = self.get_M(factor)
        self.invM = torch.inverse(self.M)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        N, C, Y, Z = x.shape
        assert C == 1
        if Z % self.factor != 0:
            x = x[..., 0 : Z // self.factor * self.factor]
        Z_new = x.shape[-1] // self.factor
        result = torch.zeros((N, C, Y, Z_new), device=x.device, dtype=x.dtype)
        for i in range(self.factor):
            result += x[..., i :: self.factor]
        result /= self.factor
        return result

    def A_T(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.factor

    def A_dagger(self, x: torch.Tensor):
        N, C, Y, Z = x.shape
        assert C == 1
        x = x.clone().detach()
        result = x.repeat_interleave(self.factor, dim=3)
        return result

    def get_M(self, factor: int):
        if factor == 2:
            return torch.tensor(
                [[0.70710678118, 0.70710678118], [0.70710678118, -0.70710678118]]
            )
        elif factor == 3:
            return torch.tensor(
                [
                    [5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                    [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                    [5.7735026e-01, 4.0824822e-01, -7.0710683e-01],
                ]
            )
        elif factor == 4:
            return torch.tensor(
                [
                    [0.5, 0.866025403784439, 0, 0],
                    [0.5, -0.288675134594813, 0.816496580927726, 0],
                    [0.5, -0.288675134594813, -0.408248290463863, 0.707106781186548],
                    [0.5, -0.288675134594813, -0.408248290463863, -0.707106781186548],
                ]
            )
        elif factor == 5:
            return torch.tensor(
                [
                    [0.447213595499958, 0.894427190999916, 0, 0, 0],
                    [0.447213595499958, -0.223606797749979, 0.866025403784439, 0, 0],
                    [0.447213595499958, -0.223606797749979, -0.288675134594813, 0.816496580927726, 0],
                    [0.447213595499958, -0.223606797749979, -0.288675134594813, -0.408248290463863, 0.707106781186548],
                    [0.447213595499958, -0.223606797749979, -0.288675134594813, -0.408248290463863, -0.707106781186548],
                ]
            )
        else:
            raise ValueError(f"unsupported zsr-factor ({factor}) for kernel")

    def _valid_width_and_pad(self, width: int):
        pad_size = width % self.factor
        valid_width = width - pad_size
        if valid_width <= 0:
            raise ValueError(
                f"Input width/depth ({width}) must be >= factor ({self.factor}) for decouple/couple."
            )
        return valid_width, pad_size

    def decouple(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape
        valid_w, w_pad_size = self._valid_width_and_pad(W)
        inputs_rs = inputs[..., :valid_w].reshape(B, C, H, valid_w // self.factor, self.factor)
        M = self.M.to(device=inputs.device, dtype=inputs.dtype)
        inputs_decp_rs = torch.einsum("bchwi,ij->bchwj", inputs_rs, M)
        inputs_decp = inputs_decp_rs.reshape(B, C, H, valid_w)
        if w_pad_size > 0:
            inputs_decp = F.pad(inputs_decp, (0, w_pad_size, 0, 0, 0, 0, 0, 0))
        return inputs_decp

    # The inverse function to `decouple`.
    def couple(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape
        valid_w, w_pad_size = self._valid_width_and_pad(W)
        inputs_rs = inputs[..., :valid_w].reshape(B, C, H, valid_w // self.factor, self.factor)
        invM = self.invM.to(device=inputs.device, dtype=inputs.dtype)
        inputs_cp_rs = torch.einsum("bchwi,ij->bchwj", inputs_rs, invM)
        inputs_cp = inputs_cp_rs.reshape(B, C, H, valid_w)
        if w_pad_size > 0:
            inputs_cp = F.pad(inputs_cp, (0, w_pad_size, 0, 0, 0, 0, 0, 0))
        return inputs_cp

    def get_mask(self, image: torch.Tensor, channel: int) -> torch.Tensor:
        B, C, H, W = image.shape
        mask = torch.zeros((B, C, H, W), device=image.device, dtype=image.dtype)
        mask[:, :, :, ::channel] = 1
        return mask