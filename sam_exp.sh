#!/bin/bash
echo 'Weak prob = 0.5'
python maml_demo.py --loss low  --labeller random --weak_prob 0.5 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot
echo 'Weak prob = 0.8'
python maml_demo.py --loss low  --labeller random --weak_prob 0.8 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot
echo ''
echo 'Weak prob = 0.9'
python maml_demo.py --loss low  --labeller random --weak_prob 0.9 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot
echo ''
echo 'Weak prob = 0.99'
python maml_demo.py --loss low  --labeller random --weak_prob 0.99 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot

