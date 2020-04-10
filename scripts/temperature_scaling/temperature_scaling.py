import torch
from torch import nn, optim
from torch.nn import functional as F
from allennlp.nn import util as nn_util


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

# This function probably should live outside of this class, but whatever
def set_temperature(model, n_layers, optimizer, data_iterator = None, cuda_device = 0):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    cuda_func = lambda x: x if cuda_device == -1 else x.cuda()
    nll_criterion = cuda_func(nn.CrossEntropyLoss())
    ece_criterion = cuda_func(_ECELoss())

    # First: collect all the logits and labels for the validation set
    if data_iterator is None:
        print("Reading {}".format(model))
        with open(model) as ifh:
            lines = [eval(x.rstrip()) for x in ifh if x[0] == '{']

        print("Read {} lines".format(len(lines)))

        labels = cuda_func(torch.LongTensor([l['label'] for l in lines]))
        logits_list = [l['logits'] for l in lines]
        all_logits = [cuda_func(torch.FloatTensor([x[layer_index] for x in logits_list])) for layer_index in range(n_layers)]
    else:
        logits_list = []
        labels_list = []
        ids_list = []
        with torch.no_grad():
            for instance in data_iterator:
                instance = nn_util.move_to_device(instance, cuda_device)
                tokens = instance['tokens']
                label = instance['label']
                instance_id = instance['instance_id']
                logits = model(tokens, -1, label)['logits']
                logits_list.append(logits)
                labels_list.append(label)
                ids_list.append(instance_id)

        labels_list = [x for _,x in sorted(zip(ids_list, labels_list), key=lambda t: t[0])]
        labels = cuda_func(torch.cat(labels_list))

        print()
        logits_list = [x for _,x in sorted(zip(ids_list, logits_list), key=lambda t: t[0])]
        ids_list = sorted(ids_list)
        for i, lo, la in zip(ids_list, logits_list, labels):
            print({"id": i[0], "logits": [x.cpu().numpy().tolist()[0] for x in lo], "label": la.item()})
    
        all_logits = [cuda_func(torch.cat([x[layer_index] for x in logits_list])) for layer_index in range(n_layers)]
        torch.save([all_logits[0], labels], 'b')

    temps = [1 for i in range(n_layers)]

    for layer_index in range(n_layers):
        logits = all_logits[layer_index]

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        best_nll = before_temperature_nll
        best_temp = 1
        for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.005, 0.0001, 0.00005, 0.00001, 0.000005]:
            temp_model = cuda_func(ModelWithTemperature(model))

            optimizer.optimize(temp_model, lr, nll_criterion, logits, labels, before_temperature_nll)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(temp_model.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(temp_model.temperature_scale(logits), labels).item()

            if after_temperature_nll < best_nll:
                best_nll = after_temperature_nll
                best_temp = temp_model.temperature.item()
                temps[layer_index] = best_temp
#                print('Optimal temperature: %.3f' % temp_model.temperature.item())
            print('lr: %f: temp: %.3f After temperature - NLL: %.3f, ECE: %.3f' % (lr, best_temp, after_temperature_nll, after_temperature_ece))

    return temps


class Optimizer():
    def __init__(self, num_epochs: int):
        self._num_epochs = num_epochs
        
    def optimize(self, temp_model: ModelWithTemperature, lr: float,
                    nll_criterion, 
                    logits: torch.FloatTensor, labels: torch.FloatTensor,
                    before_temperature_nll: float):
        raise NotImplementedError
    

class LBFGS_optimizer(Optimizer):
    def __init__(self, num_epochs: int):
        super(LBFGS_optimizer, self).__init__(num_epochs)

    def optimize(self, temp_model: ModelWithTemperature, lr: float,
                    nll_criterion, 
                    logits: torch.FloatTensor, labels: torch.FloatTensor,
                    before_temperature_nll: float):
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([temp_model.temperature], lr=lr, max_iter=self._num_epochs)

        def cal_eval():
            loss = nll_criterion(temp_model.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(cal_eval)

class Adam_optimizer(Optimizer):
    def __init__(self, num_epochs: int):
        super(Adam_optimizer, self).__init__(num_epochs)

    def optimize(self, temp_model: ModelWithTemperature, lr: float,
                    nll_criterion, 
                    logits: torch.FloatTensor, labels: torch.FloatTensor,
                    before_temperature_nll: float):

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.Adam([temp_model.temperature], lr=lr)
        best_loss = before_temperature_nll
        best_epoch = 0

        for i in range(self._num_epochs):
            temp_model.zero_grad()

            loss = nll_criterion(temp_model.temperature_scale(logits), labels)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = 1
            elif i - best_epoch > 50:
                print("Stopped at {} with value {}".format(best_epoch, best_loss))
                break


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
