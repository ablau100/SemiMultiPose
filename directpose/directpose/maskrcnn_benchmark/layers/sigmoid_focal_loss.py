import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import numpy as np

from maskrcnn_benchmark import _C
from maskrcnn_benchmark.utils.logger import setup_logger

# TODO: Use JIT to replace CUDA implementation in the future.
class _SigmoidFocalLoss(Function):
    
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = _C.sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(
            logits, targets, d_loss, num_classes, gamma, alpha
        )
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha, ground_truth=True, scores=None):
    num_classes = logits.shape[1]

    #gamma = gamma[0]
    #alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)
   
    t = targets.unsqueeze(1)
    
    if ground_truth == False:
        s = scores.unsqueeze(1)
      
    #print(torch.unique(t))
    
    p = torch.sigmoid(logits)
   # print("IS nan?? p: ", p.sum())
    print("log p: ", (torch.log(p)).sum())
    print("log 1-p: ", (torch.log(1-p)).sum())
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    print("term 1: ", term1.sum())
    print("term 2: ", term2.sum())
    
    if (np.isfinite(term1.detach().cpu().numpy()) == False).any():
        #print("LOOK")
        ind = np.isfinite(term1.detach().cpu().numpy()) == False
        ind = ind.astype(int)
        term1[ind] = torch.zeros_like(term1[ind])
        
    if (np.isfinite(term2.detach().cpu().numpy()) == False).any():
        #print("LOOK")
        ind = np.isfinite(term2.detach().cpu().numpy()) == False
        ind = ind.astype(int)
        term2[ind] = torch.zeros_like(term2[ind])
    
    if ground_truth == False:
        s = s.float().to('cuda')
        #print("s", s)
        t = t.to('cuda')
        neg_one = torch.as_tensor([-1]).float().to('cuda')
        t_n = s*(t == class_range).float()
        
        t_neg = t_n * (neg_one.expand_as(t_n))
        
        one_minus_t = torch.add(t_neg, 1).float()
        
        og_loss = -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)
        #print("og loss: ", og_loss.sum())
        les_go = -(t_n).float() * term1.float() * alpha
        les_go_2 = les_go  - (one_minus_t * (t >= 0).float()).float() * term2.float() * (1 - alpha)
        les_go_3 = les_go_2 
        print("new loss: ", les_go_3.sum())
        return les_go_3
    print("HM loss in c: ", (-(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)).sum())
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets, ground_truth=True, scores=None, from_prop=False):
       # logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
        device = logits.device
        if ground_truth == True:#logits.is_cuda:
            print("CUDAA")
            loss_func = sigmoid_focal_loss_cuda
            loss = loss_func(logits, targets, self.gamma, self.alpha)
        else:
            print("CUPUU")
            loss_func = sigmoid_focal_loss_cpu
            loss = loss_func(logits, targets, self.gamma, self.alpha, ground_truth, scores)
        #print("sig loss targets")
        #print(targets)
        #print(torch.unique(targets))
        #loss = loss_func(logits, targets, self.gamma, self.alpha, ground_truth, scores)
        print("summm: ", loss)
        if loss.sum().item() > 0:#np.isfinite(loss.sum().detach().cpu().numpy()):
            return loss.sum()
        else:
            return 0

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
