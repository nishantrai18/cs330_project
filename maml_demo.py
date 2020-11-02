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
        weakness = torch.rand((data.shape[0],)) < weak_prob
        correct = torch.rand((weakness.sum(),)) < correct_prob
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


def custom_loss_factory(loss_method, weak_coefficient):
    '''
    :param loss_method: Loss approach to choose
    '''

    loss = nn.CrossEntropyLoss(reduction='sum')

    def low_weight_loss(prediction, labels, weakness=None):
        overall_weight = 0

        if (weakness is None) or (weakness.sum() == 0):
            weak_loss = 0
            confident_loss = loss(prediction, labels)
            overall_weight = labels.shape[0]
        elif (~weakness).sum() == 0:
            weak_loss = loss(prediction, labels)
            confident_loss = 0
            overall_weight = weak_coefficient * labels.shape[0]
        else:
            weak_loss = loss(prediction[weakness], labels[weakness])
            confident_loss = loss(prediction[~weakness], labels[~weakness])
            overall_weight = ((weak_coefficient * weakness.sum()) + (~weakness).sum())

        return ((weak_coefficient * weak_loss) + confident_loss) / overall_weight

    def standard_loss(prediction, labels, weakness=None):
        return loss(prediction, labels) / labels.shape[0]

    if loss_method == StandardLoss:
        return standard_loss
    elif loss_method == LowWeightLoss:
        return low_weight_loss
    else:
        assert False, "Invalid type of loss method: {}".format(loss_method)


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
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Transform the adaptation data to become weak
    adaptation_data, adaptation_labels, adaptation_weakness = batch_processor(adaptation_data, adaptation_labels)

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels, adaptation_weakness)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    #valid_f1 = sklearn.metrics.f1_score(predictions, evaluation_labels).cpu().numpy() 
    #valid_precision = sklearn.metrics.precision_score(predictions, evaluation_labels).cpu().numpy() 
    #valid_recall = sklearn.metrics.recall_score(predictions, evaluation_labels).cpu().numpy() 
    #return valid_error, valid_accuracy, valid_f1, valid_precision
    return valid_error, valid_accuracy


def main(args, cuda=True, seed=42):

    ways = args.ways
    shots = args.shots
    meta_lr = args.meta_lr
    fast_lr = args.fast_lr
    meta_batch_size = args.meta_batch_size
    adaptation_steps = args.adaptation_steps
    num_iterations = args.num_iterations
    weak_coefficient = args.weak_coefficient
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    tasksets = l2l.vision.benchmarks.get_tasksets(
        'omniglot',
        train_ways=ways,
        train_samples=2*shots,
        test_ways=ways,
        test_samples=2*shots,
        num_tasks=20000,
        root=args.dir,
    )

    # Create model
    model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
    model.to(device)

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)

    # Define custom losses and processors
    loss = custom_loss_factory(args.loss, weak_coefficient)
    # We need dedicated train and eval processors, because we don't perform weak label generation during eval
    train_processor = batch_processor_factory(args.labeller, ways, args.weak_prob, args.correct_prob)
    eval_processor = batch_processor_factory(IdentityLabeller, ways, weak_prob=0.0)
    # NOTE: Use the one below if we want to have weak samples in the test set as well
    # eval_processor = batch_processor_factory(args.labeller, ways, args.weak_prob, args.correct_prob)

    tq = tqdm(range(num_iterations), desc="Training", position=0)

    for iteration in tq:
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
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
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               eval_processor,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()
            #meta_valid_f1 += evaluation_f1.item()
            #meta_train_precision += evaluation_precision.item()
            #meta_train_recall += evaluation_recall.item()

        # Update metrics
        tq.set_postfix({
            'iter': iteration,
            'meta-train-err': meta_train_error / meta_batch_size,
            'meta-train-acc': meta_train_accuracy / meta_batch_size,
            #'meta-train-f1': meta_train_f1 / meta_batch_size,
            # 'meta-train-precision': meta_train_precision / meta_batch_size,
            # 'meta-train-recall': meta_train_recall / meta_batch_size,
            'meta-val-err': meta_valid_error / meta_batch_size,
            'meta-val-acc': meta_valid_accuracy / meta_batch_size,
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
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', default='low', type=str, help='loss to use: low, standard')
    parser.add_argument('--labeller', default='random', type=str, help='labelling method to use: identity, random')
    parser.add_argument('--weak_prob', default=0.5, type=float, help='probability of weak samples')
    parser.add_argument('--weak_coefficient', default=0.1, type=float, help='The weight of the weak samples in the loss function')
    parser.add_argument('--correct_prob', default=0.5, type=float, help='Probability of correctness in weak samples')
    parser.add_argument('--dir', default='~/data/', type=str, help='directory to store files')

    parser.add_argument('--ways', default=5)
    parser.add_argument('--shots', default=5)
    parser.add_argument('--meta_lr', default=0.003)
    parser.add_argument('--fast_lr', default=0.5)
    parser.add_argument('--meta_batch_size', default=32)
    parser.add_argument('--adaptation_steps', default=1)
    parser.add_argument('--num_iterations', type = int, default=60000)

    args = parser.parse_args()

    main(args)
