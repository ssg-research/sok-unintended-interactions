python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.1 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.1_intgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.2 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.2_intgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.3 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.3_intgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.4 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.4_intgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.6 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.6_intgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.7 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.7_intgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.8 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.8_intgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.9 --explanations IntegratedGradients > output/ratio1_0.5_ratio_0.9_intgrad

python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.1 --explanations smoothgrad > output/ratio1_0.5_ratio_0.1_smoothgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.2 --explanations smoothgrad > output/ratio1_0.5_ratio_0.2_smoothgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.3 --explanations smoothgrad > output/ratio1_0.5_ratio_0.3_smoothgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.4 --explanations smoothgrad > output/ratio1_0.5_ratio_0.4_smoothgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.6 --explanations smoothgrad > output/ratio1_0.5_ratio_0.6_smoothgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.7 --explanations smoothgrad > output/ratio1_0.5_ratio_0.7_smoothgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.8 --explanations smoothgrad > output/ratio1_0.5_ratio_0.8_smoothgrad
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.9 --explanations smoothgrad > output/ratio1_0.5_ratio_0.9_smoothgrad

python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.1 --explanations DeepLift > output/ratio1_0.5_ratio_0.1_deeplift
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.2 --explanations DeepLift > output/ratio1_0.5_ratio_0.2_deeplift
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.3 --explanations DeepLift > output/ratio1_0.5_ratio_0.3_deeplift
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.4 --explanations DeepLift > output/ratio1_0.5_ratio_0.4_deeplift
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.6 --explanations DeepLift > output/ratio1_0.5_ratio_0.6_deeplift
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.7 --explanations DeepLift > output/ratio1_0.5_ratio_0.7_deeplift
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.8 --explanations DeepLift > output/ratio1_0.5_ratio_0.8_deeplift
python -m src.test_inference_attack --dataset CENSUS --filter sex --ratio1 0.5 --ratio2 0.9 --explanations DeepLift > output/ratio1_0.5_ratio_0.9_deeplift



python -m src.test_dimensionality --dataset CENSUS --explanations IntegratedGradients --filter sex --num_features 15 > output/numfeat_15_intgrad
python -m src.test_dimensionality --dataset CENSUS --explanations IntegratedGradients --filter sex --num_features 25 > output/numfeat_25_intgrad
python -m src.test_dimensionality --dataset CENSUS --explanations IntegratedGradients --filter sex --num_features 35 > output/numfeat_35_intgrad

python -m src.test_dimensionality --dataset CENSUS --explanations smoothgrad --filter sex --num_features 15 > output/numfeat_15_smoothgrad
python -m src.test_dimensionality --dataset CENSUS --explanations smoothgrad --filter sex --num_features 25 > output/numfeat_25_smoothgrad
python -m src.test_dimensionality --dataset CENSUS --explanations smoothgrad --filter sex --num_features 35 > output/numfeat_35_smoothgrad

python -m src.test_dimensionality --dataset CENSUS --explanations DeepLift --filter sex --num_features 15 > output/numfeat_15_deeplift
python -m src.test_dimensionality --dataset CENSUS --explanations DeepLift --filter sex --num_features 25 > output/numfeat_25_deeplift
python -m src.test_dimensionality --dataset CENSUS --explanations DeepLift --filter sex --num_features 35 > output/numfeat_35_deeplift




python -m src.test_capacity --dataset CENSUS --explanations IntegratedGradients --filter sex --model_number 1 > output/modelnum_1_intgrad
python -m src.test_capacity --dataset CENSUS --explanations DeepLift --filter sex --model_number 1 > output/modelnum_1_deeplift
python -m src.test_capacity --dataset CENSUS --explanations smoothgrad --filter sex --model_number 1 > output/modelnum_1_smoothgrad

python -m src.test_capacity --dataset CENSUS --explanations IntegratedGradients --filter sex --model_number 2 > output/modelnum_2_intgrad
python -m src.test_capacity --dataset CENSUS --explanations DeepLift --filter sex --model_number 2 > output/modelnum_2_deeplift
python -m src.test_capacity --dataset CENSUS --explanations smoothgrad --filter sex --model_number 2 > output/modelnum_2_smoothgrad

python -m src.test_capacity --dataset CENSUS --explanations IntegratedGradients --filter sex --model_number 3 > output/modelnum_3_intgrad
python -m src.test_capacity --dataset CENSUS --explanations DeepLift --filter sex --model_number 3 > output/modelnum_3_deeplift
python -m src.test_capacity --dataset CENSUS --explanations smoothgrad --filter sex --model_number 3 > output/modelnum_3_smoothgrad
