from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.fun_loss_mel_out = nn.MSELoss()
        self.fun_loss_mel_posnet_out = nn.MSELoss()
        self.fun_loss_gate = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.fun_loss_mel_out(mel_out, mel_target)
        mel_posnet_loss = self.fun_loss_mel_posnet_out(mel_out_postnet, mel_target)
        gate_loss = self.fun_loss_gate(gate_out, gate_target)
        return mel_loss + mel_posnet_loss + gate_loss
