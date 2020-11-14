#!/bin/bash
#echo 'Weak prob = 0.5'
#python maml_demo.py --loss low  --labeller random --weak_prob 0.5 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot
#echo 'Weak prob = 0.8'
#python maml_demo.py --loss low  --labeller random --weak_prob 0.8 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot
#echo ''
#echo 'Weak prob = 0.9'
#python maml_demo.py --loss low  --labeller random --weak_prob 0.9 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot
#echo ''
#echo 'Weak prob = 0.99'
#python maml_demo.py --loss low  --labeller random --weak_prob 0.99 --weak_coefficient 1.0  --num_iterations 500 --test_set_weakness True --dataset omniglot

echo 'Standard correct_prob=0.25'
python maml_demo.py --loss low --labeller random --num_iterations 2000 --correct_prob 0.25 --weak_coefficient 1 --weak_prob 0.8  --test_set_weakness True --dataset omiglot
echo ''

echo 'Standard correct_prob=0.5'
python maml_demo.py --loss low --labeller random --num_iterations 2000 --correct_prob 0.5 --weak_coefficient 1 --weak_prob 0.8  --test_set_weakness True --dataset omiglot
echo ''

echo 'Standard correct_prob=0.8'
python maml_demo.py --loss low --labeller random --num_iterations 2000 --correct_prob 0.8 --weak_coefficient 1 --weak_prob 0.8  --test_set_weakness True --dataset omiglot
echo ''

echo 'Weak correct_prob=0.25'
python maml_demo.py --loss low --labeller random --num_iterations 2000 --correct_prob 0.25 --weak_coefficient 0.4 --weak_prob 0.8  --test_set_weakness True --dataset omiglot
echo ''

echo 'Weak correct_prob=0.5'
python maml_demo.py --loss low --labeller random --num_iterations 2000 --correct_prob 0.5 --weak_coefficient 0.4 --weak_prob 0.8  --test_set_weakness True --dataset omiglot
echo ''

echo 'Weak correct_prob=0.8'
python maml_demo.py --loss low --labeller random --num_iterations 2000 --correct_prob 0.8 --weak_coefficient 0.4 --weak_prob 0.8  --test_set_weakness True --dataset omiglot
echo ''

