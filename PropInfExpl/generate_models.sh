python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split victim --filter sex --model_number 1 > logs/model1_victim_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split victim --filter sex --model_number 1  > logs/model1_victim_0.5
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split attacker --filter sex --model_number 1 > logs/model1_attacker_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split attacker --filter sex --model_number 1 > logs/model1_attacker_0.5

python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split victim --filter sex --model_number 2 > logs/model2_victim_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split victim --filter sex --model_number 2 > logs/model2_victim_0.5
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split attacker --filter sex --model_number 2 > logs/model2_attacker_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split attacker --filter sex --model_number 2 > logs/model2_attacker_0.5

python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split victim --filter sex --model_number 3 > logs/model3_victim_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split victim --filter sex --model_number 3 > logs/model3_victim_0.5
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split attacker --filter sex --model_number 3 > logs/model3_attacker_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split attacker --filter sex --model_number 3 > logs/model1_attacker_0.5

python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split victim --filter sex --model_number 4 > logs/model4_victim_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split victim --filter sex --model_number 4 > logs/model4_victim_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.1 --split attacker --filter sex --model_number 4 > logs/model4_attacker_0.1
python -m src.generate_models_capacity --dataset CENSUS --ratio 0.5 --split attacker --filter sex --model_number 4 > logs/model4_attacker_0.1





python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.1 --split victim --filter sex --num_features 35 > logs/numfeat35_victim_0.1
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.5 --split victim --filter sex --num_features 35 > logs/numfeat35_victim_0.5
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.1 --split attacker --filter sex --num_features 35 > logs/numfeat35_attacker_0.1
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.5 --split attacker --filter sex --num_features 35 > logs/numfeat35_attacker_0.5

python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.1 --split victim --filter sex --num_features 25 > logs/numfeat25_victim_0.1
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.5 --split victim --filter sex --num_features 25 > logs/numfeat25_victim_0.5
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.1 --split attacker --filter sex --num_features 25 > logs/numfeat25_attacker_0.1
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.5 --split attacker --filter sex --num_features 25 > logs/numfeat25_attacker_0.5

python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.1 --split victim --filter sex --num_features 15 > logs/numfeat15_victim_0.1
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.5 --split victim --filter sex --num_features 15 > logs/numfeat15_victim_0.5
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.1 --split attacker --filter sex --num_features 15 > logs/numfeat15_attacker_0.1
python -m src.generate_models_dimensionality --dataset CENSUS --ratio 0.5 --split attacker --filter sex --num_features 15 > logs/numfeat15_attacker_0.5






python -m src.generate_models --dataset CENSUS --ratio 0.1 --split attacker --filter sex --num 10 > logs/attacker_0.1
python -m src.generate_models --dataset CENSUS --ratio 0.2 --split attacker --filter sex --num 10 > logs/attacker_0.2
python -m src.generate_models --dataset CENSUS --ratio 0.3 --split attacker --filter sex --num 10 > logs/attacker_0.3
python -m src.generate_models --dataset CENSUS --ratio 0.4 --split attacker --filter sex --num 10 > logs/attacker_0.4
python -m src.generate_models --dataset CENSUS --ratio 0.5 --split attacker --filter sex --num 10 > logs/attacker_0.5
python -m src.generate_models --dataset CENSUS --ratio 0.6 --split attacker --filter sex --num 10 > logs/attacker_0.6
python -m src.generate_models --dataset CENSUS --ratio 0.7 --split attacker --filter sex --num 10 > logs/attacker_0.7
python -m src.generate_models --dataset CENSUS --ratio 0.8 --split attacker --filter sex --num 10 > logs/attacker_0.8
python -m src.generate_models --dataset CENSUS --ratio 0.9 --split attacker --filter sex --num 10 > logs/attacker_0.9



python -m src.generate_models --dataset CENSUS --ratio 0.1 --split victim --filter sex --num 10 > logs/victim_0.1
python -m src.generate_models --dataset CENSUS --ratio 0.2 --split victim --filter sex --num 10 > logs/victim_0.2
python -m src.generate_models --dataset CENSUS --ratio 0.3 --split victim --filter sex --num 10 > logs/victim_0.3
python -m src.generate_models --dataset CENSUS --ratio 0.4 --split victim --filter sex --num 10 > logs/victim_0.4
python -m src.generate_models --dataset CENSUS --ratio 0.5 --split victim --filter sex --num 10 > logs/victim_0.5
python -m src.generate_models --dataset CENSUS --ratio 0.6 --split victim --filter sex --num 10 > logs/victim_0.6
python -m src.generate_models --dataset CENSUS --ratio 0.7 --split victim --filter sex --num 10 > logs/victim_0.7
python -m src.generate_models --dataset CENSUS --ratio 0.8 --split victim --filter sex --num 10 > logs/victim_0.8
python -m src.generate_models --dataset CENSUS --ratio 0.9 --split victim --filter sex --num 10 > logs/victim_0.9