python -m src.inversion_attack > output/interaction_type

python -m src.test_model_size --modelnum 1 > output/modelnum_1
python -m src.test_model_size --modelnum 2 > output/modelnum_2
python -m src.test_model_size --modelnum 3 > output/modelnum_3
python -m src.test_model_size --modelnum 4 > output/modelnum_4



python -m src.test_input_size --input_size 10 > output/input_size_10
python -m src.test_input_size --input_size 20 > output/input_size_20
python -m src.test_input_size --input_size 30 > output/input_size_30
python -m src.test_input_size --input_size 40 > output/input_size_40