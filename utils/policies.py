import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        _, action_dim = out.size()
        # dim(u_aaction)=5, dim(r_action) = 2, dim(audio_action = 3)
        r_action_dim = 2
        audio_action_dim = 3
        u_action_dim = action_dim - (r_action_dim + audio_action_dim)
        # print(action_dim == 5 + r_action_dim + audio_action_dim)
        assert action_dim == 5 + r_action_dim + audio_action_dim, "policy dimensions"

        probs_u = F.softmax(out[:,0:u_action_dim], dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act_u = categorical_sample(probs_u, use_cuda=on_gpu)
        else:
            act_u = onehot_from_logits(probs_u)

        # TODO: change rotation to discrete action, and output prob_r, also change the step in environment
        # action_r = out[:, u_action_dim].view(-1, 1)
        probs_r = F.softmax(out[:, u_action_dim:u_action_dim+r_action_dim], dim=1)
        # on_gpu = next(self.parameters()).is_cuda
        if sample:
            _, act_r = categorical_sample(probs_r, use_cuda=on_gpu)
        else:
            act_r = onehot_from_logits(probs_r)

        probs_audio = F.softmax(out[:, u_action_dim+r_action_dim:], dim=1)
        # on_gpu = next(self.parameters()).is_cuda
        if sample:
            _, act_audio = categorical_sample(probs_audio, use_cuda=on_gpu)
        else:
            act_audio = onehot_from_logits(probs_audio)

        return torch.cat([act_u, act_r, act_audio], dim=1)


        # probs = F.softmax(out, dim=1)
        # on_gpu = next(self.parameters()).is_cuda
        # if sample:
        #     int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        # else:
        #     act = onehot_from_logits(probs)
        # rets = [act]
        # if return_log_pi or return_entropy:
        #     log_probs = F.log_softmax(out, dim=1)
        # if return_all_probs:
        #     rets.append(probs)
        # if return_log_pi:
        #     # return log probability of selected action
        #     rets.append(log_probs.gather(1, int_act))
        # if regularize:
        #     rets.append([(out**2).mean()])
        # if return_entropy:
        #     rets.append(-(log_probs * probs).sum(1).mean())
        # if len(rets) == 1:
        #     return rets[0]
        # return rets
