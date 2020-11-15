#!/bin/bash
echo 'proto correct_prob=0.0'
python maml_demo.py --loss proto --labeller random --num_iterations 1000 --correct_prob 0.0 --weak_coefficient 0.4 --weak_prob 0.8  --dataset omniglot --test_set_weakness True  --features_to_use fets
echo ''

echo 'proto correct_prob=0.25'
python maml_demo.py --loss proto --labeller random --num_iterations 1000 --correct_prob 0.25 --weak_coefficient 0.4 --weak_prob 0.8  --dataset omniglot --test_set_weakness True  --features_to_use fets
echo ''

echo 'proto correct_prob=0.5'
python maml_demo.py --loss proto --labeller random --num_iterations 1000 --correct_prob 0.5 --weak_coefficient 0.4 --weak_prob 0.8  --dataset omniglot --test_set_weakness True  --features_to_use fets
echo ''

echo 'proto correct_prob=0.8'
python maml_demo.py --loss proto --labeller random --num_iterations 1000 --correct_prob 0.8 --weak_coefficient 0.4 --weak_prob 0.8  --dataset omniglot --test_set_weakness True  --features_to_use fets
echo ''

