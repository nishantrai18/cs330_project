#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import argparse
import random
import numpy as np
import torch
import learn2learn as l2l
import sklearn

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 

from custom_models import CustomOmniglotFC, CustomMiniImagenetCNN

from torch import nn, optim
from tqdm import tqdm


RandomLabeller = "random"
IdentityLabeller = "identity"


def batch_processor_factory(labelling_method, ways, weak_prob, correct_prob=0.5):
    '''
    :param labelling_method: Which labelling method to use
    :param weak_prob: Probability of generating weak label
    :return: Generator functions to perform the processing
    '''

    def random_label_generator(data, labels):
        '''
        Returns data, labels, is_weakly_labelled
        '''
        # TODO: Ensure that we have at least one correct sample for each class
        # TODO: Remove randomness from here, let's have fixed ratios instead, makes the code simpler
        #weakness = torch.rand((data.shape[0],)) < weak_prob
        weakness = ~(torch.rand((data.shape[0],)) < 1) # make a tensor of False
        idx = torch.randperm(weakness.size(0))[:int(weakness.size(0)*weak_prob)] # this will select exactly weak_prob percent
        weakness[idx] = True # mark which indexes will be weak
        
        # ENSURE THAT NO MATTER WHAT (N WAYS) EXAMPLES WILL BE TRUE - so there will be at least one correct example from each class
        # WE FORCE THE FIRST SAMPLE IN EACH CLASS TO BE TRUE
        label2firstidx = {}
        i = 0
        for lb in labels:
            if lb.item() not in label2firstidx:
                label2firstidx[lb.item()] = i
            i+=1
        idx = list(label2firstidx.values())
        weakness[idx] = False # not weak means it is marked as correct
        
        #correct = torch.rand((weakness.sum(),)) < correct_prob
        correct = ~(torch.rand((weakness.sum(),)) < 1) # make a tensor of False
        idx = torch.randperm(correct.size(0))[:int(correct.size(0)*correct_prob)] # this will select exactly correct_prob percent to be marked as true of the weak data 
        correct[idx] = True # mark which indexes of the weakness will be marked as correct

        labels[weakness][~correct] = torch.randint(0, ways, (labels[weakness][~correct].shape[0],), device=labels.device)
        return data, labels, weakness

    def identity_generator(data, labels):
        '''
        Does nothing - useless but needed for compatibility
        '''
        weakness = torch.zeros((data.shape[0],)) == 1
        return data, labels, weakness

    if labelling_method == IdentityLabeller:
        return identity_generator
    elif labelling_method == RandomLabeller:
        return random_label_generator
    else:
        assert False, "Invalid type of labelling method: {}".format(labelling_method)


StandardLoss = "standard"
LowWeightLoss = "low"
ProtoLabelLoss = "proto"
temp = 1

def custom_loss_factory(loss_method, ways, weak_coefficient):
    '''
    :param loss_method: Loss approach to choose
    '''

    loss = nn.CrossEntropyLoss(reduction='sum')

    def categorical_cross_entropy(y_pred, y_true):
        y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
        return -(y_true * torch.log(y_pred)).sum(dim=1).sum()

    proto_loss = categorical_cross_entropy

    def low_weight_loss(prediction, labels, weakness, _=None):
        stats = {}

        stats['nweak'] = weakness.sum()
        stats['nconf'] = labels.shape[0] - stats['nweak']

        if stats['nweak'] == 0:
            stats['lweak'] = 0
            stats['lconf'] = loss(prediction, labels)
        elif stats['nconf'] == 0:
            stats['lweak'] = loss(prediction, labels)
            stats['lconf'] = 0
        else:
            stats['lweak'] = loss(prediction[weakness], labels[weakness])
            stats['lconf'] = loss(prediction[~weakness], labels[~weakness])

        return stats

    def proto_label_loss(prediction, labels, weakness, etc=None):
        '''
        Returns the loss for confident samples
        For weak samples,
            - First compute predictions based on clustering
            - Then proceed to compute standard cross entropy loss between the predictions and clustering based approach
        '''
        stats = {}

        stats['nweak'] = weakness.sum()
        stats['nconf'] = labels.shape[0] - stats['nweak']

        if stats['nweak'] == 0:
            stats['lweak'] = 0
            stats['lconf'] = loss(prediction, labels)
        elif stats['nconf'] == 0:
            stats['lweak'] = loss(prediction, labels)
            stats['lconf'] = 0
        else:
            weak_labels = labels[weakness]
            # Potential experiment: Add weak label vs not adding it here; results will change as we change the
            # correct ratio in the weak labels
            features = etc['features']
            clustered_label_probs = compute_protonet_like_labels(
                features[~weakness], features[weakness], labels[~weakness], num_classes=ways, temperature=temp)
            # Compute loss
            stats['lweak'] = proto_loss(prediction[weakness], clustered_label_probs)
            stats['lconf'] = loss(prediction[~weakness], labels[~weakness])
            stats['wacc'] = \
                (prediction[weakness].argmax(dim=-1) == etc['gt_labels'][weakness]).sum() * 1.0 / (labels[weakness].shape[0] * 1.0)

        return stats

    def standard_loss(prediction, labels, weakness, _=None):
        # Modify the global weak weight as standard is just low_weight with coeff 1
        global weak_coefficient
        weak_coefficient = 1.0
        return low_weight_loss(prediction, labels, weakness)

    def loss_wrapper(loss_fn):

        def wrapper(prediction, labels, weakness=None, etc=None):
            if weakness is None:
                weakness = torch.zeros(labels.shape[0])

            stats = loss_fn(prediction, labels, weakness, etc)
            overall_weight = (stats['nconf']) + (stats['nweak'] * weak_coefficient)
            overall_loss = (stats['lconf']) + (stats['lweak'] * weak_coefficient)
            stats['avg_lconf'] = stats['lconf'] / stats['nconf']
            stats['avg_lweak'] = stats['lweak'] / stats['nweak']
            stats['overall_loss'] = overall_loss / overall_weight
            return stats

        return wrapper

    def get_loss_method():
        if loss_method == StandardLoss:
            return standard_loss
        elif loss_method == LowWeightLoss:
            return low_weight_loss
        elif loss_method == ProtoLabelLoss:
            return proto_label_loss
        else:
            assert False, "Invalid type of loss method: {}".format(loss_method)

    return loss_wrapper(get_loss_method())


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, batch_processor, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, gt_adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Transform the adaptation data to become weak
    adaptation_data, adaptation_labels, adaptation_weakness = batch_processor(adaptation_data, gt_adaptation_labels)

    stats = {}
    # Adapt the model
    for step in range(adaptation_steps):
        logits, features = learner(adaptation_data)
        stats = loss(
            logits,
            adaptation_labels,
            adaptation_weakness,
            {'features': features, 'gt_labels': gt_adaptation_labels})
        learner.adapt(stats['overall_loss'])

    # Evaluate the adapted model
    predictions, _ = learner(evaluation_data)
    valid_stats = loss(predictions, evaluation_labels)
    valid_error = valid_stats['overall_loss']
    valid_accuracy = accuracy(predictions, evaluation_labels)
    #valid_f1 = sklearn.metrics.f1_score(predictions, evaluation_labels).cpu().numpy() 
    #valid_precision = sklearn.metrics.precision_score(predictions, evaluation_labels).cpu().numpy() 
    #valid_recall = sklearn.metrics.recall_score(predictions, evaluation_labels).cpu().numpy() 
    #return valid_error, valid_accuracy, valid_f1, valid_precision

    # process stats to be detached cpu floats
    stats = {k: v.cpu().detach().item() for k, v in stats.items()}

    return valid_error, valid_accuracy, stats


def pairwise_distance(A, B):
    # squared norms of each row in A and B
    na = torch.sum(A * A, dim=1)
    nb = torch.sum(B * B, dim=1)

    # na as a row and nb as a column vectors
    na = na.reshape([-1, 1])
    nb = nb.reshape([1, -1])

    # return pairwise euclidean difference matrix
    D = torch.sqrt(torch.clamp(na - 2 * A.matmul(B.transpose(1, 0)) + nb, min=0.0))

    return D


def get_prototypes_from_labels(samples, labels):
    '''
    Returns a N x D matrix where ith row represents ith class
    '''
    N = samples.shape[0]
    M = torch.zeros(labels.max() + 1, N)
    M[labels, torch.arange(N)] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1).to(samples.device)
    return torch.mm(M, samples)


def compute_protonet_like_labels(support, q_latent, support_labels, num_classes, weak_query_labels=None,temperature = 1.0):
    """
      calculates the prototype network like proposed labels using the latent representation of x
      and the latent representation of the queries
      Args:
        support: latent representation of supports with known labels with shape [S, D], where D is the latent dimension
        q_latent: latent representation of queries with shape [Q, D], where D is the latent dimension
        support_labels: Labels of the supports with shape [S]
        weak_query_labels: Weak labels for the query, optional
      Returns:
        proposed_labels: predicted labels for the queries
    """

    assert torch.unique(support_labels).shape[0] == num_classes, "Invalid num classes: {}".format(
        torch.unique(support_labels)
    )

    prototypes = get_prototypes_from_labels(support, support_labels)
    distance = pairwise_distance(q_latent, prototypes)

    # Another hyperparameter which can be tuned, makes the label smoother and allows for more room for mistakes in
    # the proposal
    #temperature = 1.0
    logits = -distance
    probabilities = torch.nn.functional.softmax(logits / temperature, dim=1)

    if weak_query_labels is not None:
        # Another hyper-parameter to tune, MIGHT be able to learn this as well
        balance = 0.5
        weak_query_probs = torch.eye(support_labels.max() + 1)[weak_query_labels]
        probabilities = ((probabilities * balance) + (weak_query_probs * (1 - balance)))

    return probabilities


def get_model_for_dataset(args):
    # Create model
    model = None
    features_to_use = args.features_to_use
    if args.dataset == 'omniglot':
        model = CustomOmniglotFC(28 ** 2, args.ways, return_fets=features_to_use)
    elif args.dataset == 'mini-imagenet':
        model = CustomMiniImagenetCNN(args.ways, return_fets=features_to_use)
    model.to(args.device)

    return model


def main(args, cuda=True, seed=42):

    ways = args.ways
    shots = args.shots
    meta_lr = args.meta_lr
    fast_lr = args.fast_lr
    meta_batch_size = args.meta_batch_size
    adaptation_steps = args.adaptation_steps
    num_iterations = args.num_iterations
    weak_coefficient = args.weak_coefficient
    test_set_weakness = args.test_set_weakness
    temperature = args.temperature
    global temp 
    temp = temperature
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    args.device = device

    tasksets = l2l.vision.benchmarks.get_tasksets(
        args.dataset,
        train_ways=ways,
        train_samples=2*shots,
        test_ways=ways,
        test_samples=2*shots,
        num_tasks=20000,
        root=args.dir,
    )

    model = get_model_for_dataset(args)

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)

    # Define custom losses and processors
    loss = custom_loss_factory(args.loss, args.ways, weak_coefficient)
    # We need dedicated train and eval processors, because we don't perform weak label generation during eval
    train_processor = batch_processor_factory(args.labeller, ways, args.weak_prob, args.correct_prob)
    
    if test_set_weakness:
        eval_processor = batch_processor_factory(args.labeller, ways, args.weak_prob, args.correct_prob)
    # NOTE: Use the one below if we want to have weak samples in the test set as well
    else:
        eval_processor = batch_processor_factory(IdentityLabeller, ways, weak_prob=0.0)

    tq = tqdm(range(num_iterations), desc="Training", position=0)

    for iteration in tq:
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        proto_train_acc = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy, train_stats = fast_adapt(batch,
                                                               train_processor,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()
            #meta_train_f1 += evaluation_f1.item()
            #meta_train_precision += evaluation_precision.item()
            #meta_train_recall += evaluation_recall.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy, val_stats = fast_adapt(batch,
                                                               eval_processor,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()
            proto_train_acc += train_stats['wacc']
            #meta_valid_f1 += evaluation_f1.item()
            #meta_train_precision += evaluation_precision.item()
            #meta_train_recall += evaluation_recall.item()

        # Update metrics, legend is here
        # mt - meta train
        # mv - meta val
        # lconf - confident loss i.e. for strongly labeled data
        # lweak - weak loss i.e. for weakly labeled data
        tq.set_postfix({
            'mt-err': meta_train_error / meta_batch_size,
            'mt-acc': meta_train_accuracy / meta_batch_size,
            'mt-lconf': train_stats['lconf'],
            'mt-lweak': train_stats['lweak'],
            'wacc': proto_train_acc / meta_batch_size,
            #'meta-train-f1': meta_train_f1 / meta_batch_size,
            # 'meta-train-precision': meta_train_precision / meta_batch_size,
            # 'meta-train-recall': meta_train_recall / meta_batch_size,
            'mv-err': meta_valid_error / meta_batch_size,
            'mv-acc': meta_valid_accuracy / meta_batch_size,
            'mv-lconf': val_stats['lconf'],
            'mv-lweak': val_stats['lweak'],
            #'meta-val-f1': meta_valid_f1 / meta_batch_size,
            #'meta-val-precision': meta_valid_precision / meta_batch_size,
            #'meta-val-recall': meta_valid_recall / meta_batch_size,
        })

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy, test_stats = fast_adapt(batch,
                                                           eval_processor,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


def run_unittests():
    A = torch.tensor([[0, 1, 0], [1, 0, 1]]).float()
    B = torch.tensor([[1, 1, 0], [0, 1, 1]]).float()
    labelsA = torch.tensor([[0, 1]])
    labelsB = torch.tensor([[1, 0]])

    print(pairwise_distance(A, A))
    print(pairwise_distance(A, B))
    print(pairwise_distance(B, A))
    print(compute_protonet_like_labels(A, B, labelsA, 2, weak_query_labels=None))
    print(compute_protonet_like_labels(A, B, labelsA, 2, weak_query_labels=labelsB))


def str2bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--unittest', default=False, type=str2bool, help='override and run unittest')

    parser.add_argument('--labeller', default='random', type=str, help='labelling method to use: identity, random')
    parser.add_argument('--weak_prob', default=0.2, type=float, help='probability of weak samples')
    parser.add_argument('--correct_prob', default=0.5, type=float, help='Probability of correctness in weak samples')
    parser.add_argument('--dir', default='~/data/', type=str, help='directory to store files')
    parser.add_argument('--dataset', default='omniglot', type=str, help='dataset to use')
    parser.add_argument('--test_set_weakness', default=True, type=bool, help='Makes test set with weak labels in its')

    # Flags related to the approach we're using
    parser.add_argument('--loss', default='low', type=str, help='Which approach to use: standard, low, proto')
    parser.add_argument('--weak_coefficient', default=0.1, type=float, help='The weight of the weak samples in the loss function')

    parser.add_argument('--ways', default=5, type=int)
    parser.add_argument('--shots', default=5, type=int)
    parser.add_argument('--meta_lr', default=0.003)
    parser.add_argument('--fast_lr', default=0.5)
    parser.add_argument('--meta_batch_size', default=32)
    parser.add_argument('--adaptation_steps', default=1)
    parser.add_argument('--num_iterations', type=int, default=60000)
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--features_to_use', default='logits', type=str, help='Choose one of input, fets')

    args = parser.parse_args()

    if args.unittest:
        run_unittests()
    else:
        main(args)
