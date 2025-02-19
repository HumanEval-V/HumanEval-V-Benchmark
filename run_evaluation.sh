exp_base_dir="output/example_exp"

model_names=("gpt_4o" )
exp_types=("V2C" "V2C-CoT" "V2T2C" "V2T2C-4o" "GT-T2C")
sample_num=(1 6)
for model_name in "${model_names[@]}"; do
    for exp_type in "${exp_types[@]}"; do
        for num in "${sample_num[@]}"; do
            echo " "
            echo "model_name: $model_name, exp_type: $exp_type, sample_num: $num"
            python inference.py --model_name $model_name --exp_type $exp_type --sample_num $num --exp_base_dir $exp_base_dir
            python evaluate.py --model_name $model_name --exp_type $exp_type --sample_num $num --exp_base_dir $exp_base_dir
        done
    done
done