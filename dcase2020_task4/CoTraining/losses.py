import torch
import torch.nn as nn


def weak_synth_loss(logits_weak, logits_strong, y_weak, y_strong, reduce: str = "mean"):
    assert reduce in ["mean", "sum"], "support only \"mean\" and \"sum\""
    
    #  Reduction function
    if reduce == "mean":
        reduce_fn = torch.mean
    elif reduce == "sum":
        reduce_fn = torch.sum
    
    # based on Binary Cross Entropy loss
    weak_criterion = nn.BCEWithLogitsLoss(reduction="none")
    strong_criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    # calc separate loss function
    weak_bce = weak_criterion(logits_weak, y_weak)
    strong_bce = strong_criterion(logits_strong, y_strong)
    
    weak_bce = reduce_fn(weak_bce, dim=1)
    strong_bce = reduce_fn(strong_bce, dim=(1, 2))
    
    # calc strong mask
    strong_mask = torch.clamp(torch.sum(y_strong, dim=(1, 2)), 0, 1) # vector of 0 or 1
#     strong_mask = strong_mask.detach() # declared not to need gradients

    
    # Output the different loss for logging purpose
    weak_loss = reduce_fn(weak_bce)
    strong_loss = reduce_fn(strong_mask * strong_bce)
    total_loss = reduce_fn(weak_bce + strong_mask * strong_bce)
    
    return weak_loss, strong_loss, total_loss


def loss_sup(logits_S1, logit_S2, labels_S1, labels_S2):
    loss1 = weak_synth_loss(logit_S1, labels_S1)
    loss2 = weak_synth_loss(logit_S2, labels_S2)
    return (loss1 + loss2)


def loss_cot(U_p1, U_p2):
    # the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Sigmoid()
    LS = nn.LogSigmoid()
    U_batch_size = U_p1.size()[0]

    a1 = 0.5 * (S(U_p1) + S(U_p2))
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)

    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)

    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3)) / U_batch_size


def loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2,
              logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2):
    S = nn.Sigmoid()
    LS = nn.LogSigmoid()

    S_batch_size = logit_S1.size()[0]
    U_batch_size = logit_U1.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = S(logit_S2) * LS(perturbed_logit_S1)
    b = S(logit_S1) * LS(perturbed_logit_S2)
    c = S(logit_U2) * LS(perturbed_logit_U1)
    d = S(logit_U1) * LS(perturbed_logit_U2)
        
    a = torch.sum(a)
    b = torch.sum(b)
    c = torch.sum(c)
    d = torch.sum(d)

    return -(a + b + c + d) / total_batch_size


def p_loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2,
                logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2):
    S = nn.Sigmoid()
    LS = nn.LogSigmoid()

    S_batch_size = logit_S1.size()[0]
    U_batch_size = logit_U1.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = S(logit_S2) * LS(perturbed_logit_S1)
    a = torch.sum(a)

    b = S(logit_S1) * LS(perturbed_logit_S2)
    b = torch.sum(b)

    c = S(logit_U2) * LS(perturbed_logit_U1)
    c = torch.sum(c)

    d = S(logit_U1) * LS(perturbed_logit_U2)
    d = torch.sum(d)

    pld_S = -(a + b) / S_batch_size
    pld_U = -(c + d) / U_batch_size
    ldiff = -(a + b + c + d) / total_batch_size

    return pld_S, pld_U, ldiff
 