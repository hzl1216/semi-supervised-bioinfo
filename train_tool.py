
import time
import os
from set_args import create_parser
from util.net import WideResNet,TCN, ResNet50
from util import ramps
from util.losses import *
from torch.autograd import  Variable
import logging
from util.utils import *
from util.losses import entropy_loss
import util.dataset as dataset
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix
NO_LABEL=dataset.NO_LABEL
LOG = logging.getLogger('main')
args =None
def set_args(input_args):
    global args
    args = input_args




def train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model, optimizer,ema_optimizer, epoch,scheduler=None):
    labeled_train_iter = iter(train_labeled_loader)
    unlabeled_train_iter = iter(train_unlabeled_loader)
    class_criterion = nn.CrossEntropyLoss().cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, args.consistency_type

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i in range(args.epoch_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_labeled_loader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            inputs_u, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(train_unlabeled_loader)
            inputs_u,  _ = unlabeled_train_iter.next()

        # measure data loading time
        meters.update('data_time', time.time() - end)


        inputs_x = inputs_x.cuda()
        batch_size = inputs_x.size(0)
        targets_x_onehot= torch.zeros(batch_size, 33).scatter_(1, targets_x.view(-1,1), 1)
        targets_x = targets_x.cuda(non_blocking=True)
        
        inputs_u = inputs_u.cuda()

        with torch.no_grad():

            ema_outputs_u = ema_model(inputs_u)
            if args.mixup:
                ema_outputs_u = sharpen(ema_outputs_u)
            ema_outputs_u = ema_outputs_u.detach()
        if args.mixup:
            targets_x = targets_x_onehot.cuda()
            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([targets_x, ema_outputs_u], dim=0)
            outputs_x,outputs_u,targets_x,targets_u = mixup(all_inputs,all_targets,batch_size,model)
            loss,class_loss ,consistency_loss = semiloss_mixup(outputs_x, targets_x, outputs_u, targets_u,epoch)
        else:
            outputs_x = model(inputs_x)
            outputs_u = model(inputs_u)
            loss,class_loss ,consistency_loss = semiLoss(outputs_x, targets_x, outputs_u, ema_outputs_u, class_criterion, consistency_criterion,epoch)

        meters.update('loss', loss.item())
        meters.update('class_loss', class_loss.item())
        meters.update('cons_loss', consistency_loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        ema_optimizer.step(bn=True)
        if scheduler is not None:
            scheduler.step()


        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'.format(
                    epoch, i, args.epoch_iteration, meters=meters))
#    ema_optimizer.step(bn=True)
    return meters.averages()['class_loss/avg'],meters.averages()['cons_loss/avg']


def train(train_labeled_loader, model, ema_model, optimizer, ema_optimizer, epoch,
               scheduler=None):
    labeled_train_iter = iter(train_labeled_loader)
    class_criterion = nn.CrossEntropyLoss().cuda()
    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i in range(args.epoch_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_labeled_loader)
            inputs_x, targets_x = labeled_train_iter.next()

        # measure data loading time
        meters.update('data_time', time.time() - end)

        inputs_x = inputs_x.cuda()
        batch_size = inputs_x.size(0)
        targets_x_onehot = torch.zeros(batch_size, 33).scatter_(1, targets_x.view(-1, 1), 1)
        targets_x = targets_x.cuda(non_blocking=True)

        if args.mixup:
            targets_x = targets_x_onehot.cuda()
            all_inputs = torch.cat([inputs_x], dim=0)
            all_targets = torch.cat([targets_x], dim=0)
            l = np.random.beta(args.alpha, args.alpha)

            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b
            mixed_outputs = model(mixed_input)
            loss = -torch.mean(torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_target, dim=1))
        else:
            outputs_x = model(inputs_x)

            loss=class_criterion(outputs_x, targets_x)

        meters.update('loss', loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        ema_optimizer.step(bn=True)
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'loss {meters[loss]:.4f}\t'.format(
                    epoch, i, args.epoch_iteration, meters=meters))
    #    ema_optimizer.step(bn=True)
    return meters.averages()['loss/avg']

def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    all_labels = None
    all_outputs = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if all_labels is None:
                all_labels = targets
                all_outputs = outputs
            else:
                all_labels = torch.cat([all_labels,targets],dim=0)
                all_outputs = torch.cat([all_outputs,outputs],dim=0)


            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if batch_idx % args.print_freq == 0:
                print('{batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
    conf_matrix = confusion_matrix(all_outputs, all_labels)
    plot_confusion_matrix(conf_matrix.numpy(),epoch)
    return losses.avg, top1.avg


class WeightEMA(object):
    def __init__(self, model, ema_model, tmp_model=None, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        if tmp_model is not None:
            self.tmp_model = tmp_model.cuda()
#        self.tmp_model =  Full_net(9964,33).cuda()
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                param.data.mul_(1 - args.weight_decay)



def update_ema_variables(model, ema_model,epoch, alpha):
    # Use the true average until the exponential average is more correct
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def save_checkpoint(state,  dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_entropy_weight(epoch):
    if epoch>args.consistency_rampup:
            
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return  args.entropy_cost 
    else:
        return 0


def sharpen(outputs):
    outputs = torch.softmax(outputs, dim=1)
    outputs = outputs**2
    outputs  = outputs / outputs.sum(dim=1, keepdim=True)
    return outputs


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def mixup(all_inputs,all_targets,batch_size,model):
    l = np.random.beta(args.alpha, args.alpha)

    l = max(l, 1 - l)

    idx = torch.randperm(all_inputs.size(0))

    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
    mixed_input = list(torch.split(mixed_input, batch_size))
    mixed_input = interleave(mixed_input, batch_size)

    logits = [model(mixed_input[0])]
    for input in mixed_input[1:]:
        logits.append(model(input))

    # put interleaved samples back
    logits = interleave(logits, batch_size)
    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)
    return logits_x,logits_u, mixed_target[:batch_size], mixed_target[batch_size:]

def semiLoss(outputs_x, targets_x, outputs_u, targets_u, class_criterion, consistency_criterion,epoch):
    class_loss = class_criterion(outputs_x,targets_x)
    consistency_loss = consistency_criterion(outputs_u,targets_u)
    consistency_weight = args.consistency*ramps.linear_rampup(epoch, args.epochs)

    return class_loss + consistency_weight*consistency_loss, class_loss, consistency_loss*consistency_weight


def semiloss_mixup(outputs_x, targets_x, outputs_u, targets_u, epoch):
    probs_u = torch.softmax(outputs_u, dim=1)
    class_loss = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    consistency_loss = torch.mean((probs_u - targets_u)**2)
#    consistency_loss = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))
    consistency_weight = args.consistency*ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    return class_loss + consistency_weight*consistency_loss, class_loss, consistency_loss*consistency_weight



def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def confusion_matrix(preds, labels, n_class=33):
    conf_matrix= torch.zeros(n_class, n_class)
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix




