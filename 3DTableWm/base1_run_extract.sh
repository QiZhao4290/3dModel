#!/bin/bash
exp_tag='testing'
isClassify_oridata=false
isEmbed=false
isExtract_no_attack=true
isClassify_wmdata=false
isNoiseAttack=false
isDeleteAttack=false
isInsertAttack=false
isClassify=false
isExtract=false
isPlot=false

csv_dir="./base1_dataset_tobe_wm"
clean_file="$csv_dir/table_tobe_wm.csv"

method="NR"
dataset="3Dmodel"
test_set="./base1_covtype_standardized_test.csv"
remain_file="./base1_covtype_standardized_remain.csv"
exp_root="./base1_experiments"
#eigenVectors_mat='./eigenVectors_covtype_Mar26.mat'
classify_res_folder="$exp_root/$exp_tag/$dataset/classification_results"
classify_res_file="classification_results_$exp_tag.csv"
wm_params_json="./base1_params.json"
wm_datasets_folder="$exp_root/$exp_tag/$dataset/embed/wm_datasets"
keys_folder="$exp_root/$exp_tag/$dataset/embed/storedkeys"

if $isClassify_oridata; then
  python base1_classification_channel/base1_classification.py --target_dir "$csv_dir" --test_set "$test_set" --result_dir "$classify_res_folder" --result_file "$classify_res_file" --dataset "$dataset" --method "$method"
fi

if $isEmbed; then
  python base1_embed.py --org_file "$clean_file" --wm_params_json "$wm_params_json"  --exp_root "$exp_root" --exp_tag "$exp_tag" --dataset "$dataset"
fi

if $isClassify_wmdata; then
  python base1_classification_channel/base1_classification.py --target_dir "$wm_datasets_folder" --test_set "$test_set" --result_dir "$classify_res_folder" --result_file "$classify_res_file" --dataset "$dataset" --method "$method"
fi

if $isExtract_no_attack; then
  python base1_extract.py --exp_root "$exp_root" --exp_tag "$exp_tag" --target_dir "embed" --dataset "$dataset" --attack "watermarked" --add_info "no_attack"
fi

if $isNoiseAttack; then
  # noise_type_list=("uniform")
  noise_type_list=("rand" "gaussian" "uniform")
  for noise_type in "${noise_type_list[@]}"
  do
    target_dir_noise="$exp_root/$exp_tag/$dataset/noise/$noise_type"
    python base1_attack_channel/base1_noise_attack.py --clean_file "$clean_file" --wm_folder "$wm_datasets_folder" --output_dir "$target_dir_noise" --noise_type "$noise_type" --method "$method"

    if $isClassify; then
      python base1_classification_channel/base1_classification.py --target_dir "$target_dir_noise" --test_set "$test_set" --result_dir "$classify_res_folder" --result_file "$classify_res_file" --dataset "$dataset" --method "$method"
    fi

    if $isExtract; then
      python base1_extract.py --exp_root "$exp_root" --exp_tag "$exp_tag" --dataset "$dataset" --attack "noise" --add_info "$noise_type" --target_dir "$target_dir_noise"
    fi
  done
fi


if $isDeleteAttack; then
  delete_type_list=("row")
  for delete_type in "${delete_type_list[@]}"
  do
    target_dir_delete="$exp_root/$exp_tag/$dataset/delete/$delete_type"
    python base1_attack_channel/base1_delete_attack.py --clean_file "$clean_file" --wm_folder "$wm_datasets_folder" --output_dir "$target_dir_delete" --delete_type "$delete_type" --method "$method" --dataset "$dataset"

    if $isClassify; then
      python base1_classification_channel/base1_classification.py --target_dir "$target_dir_delete" --test_set "$test_set" --result_dir "$classify_res_folder" --result_file "$classify_res_file" --dataset "$dataset" --method "$method"
    fi

    if $isExtract; then
      python base1_extract.py --exp_root "$exp_root" --exp_tag "$exp_tag" --dataset "$dataset" --attack "delete" --add_info "$delete_type" --target_dir "$target_dir_delete"
    fi
  done
fi


if $isInsertAttack; then
  insert_type_list=("concatenate")
  for insert_type in "${insert_type_list[@]}"
  do
    target_dir_insert="$exp_root/$exp_tag/$dataset/insert/$insert_type"
    python base1_attack_channel/base1_insert_attack.py --clean_file "$clean_file" --wm_folder "$wm_datasets_folder" --remain_file "$remain_file" --output_dir "$target_dir_insert" --insert_type "$insert_type" --method "$method" --dataset "$dataset"

    if $isClassify; then
      python base1_classification_channel/base1_classification.py --target_dir "$target_dir_insert" --test_set "$test_set" --result_dir "$classify_res_folder" --result_file "$classify_res_file" --dataset "$dataset" --method "$method"
    fi

    if $isExtract; then
      python base1_extract.py --exp_root "$exp_root" --exp_tag "$exp_tag" --dataset "$dataset" --attack "insert" --add_info "$insert_type" --target_dir "$target_dir_insert"
    fi
  done
fi


echo "==> start plotting"
if $isPlot; then
  extract_root="$exp_root/$exp_tag/$dataset/extract"
  record_folder="$exp_root/$exp_tag/$dataset/extract_results_sum"
  sum_constant=12
  echo 'sum_constant = 12'
  python base1_utils/base1_plot_extract_res.py --exp_root_nr "$extract_root" --save_folder "$record_folder" --attack "watermarked" --add_info "no_attack" --sum_constant $sum_constant
  python base1_utils/base1_plot_extract_res.py --exp_root_nr "$extract_root" --save_folder "$record_folder" --attack "noise" --add_info "uniform" --sum_constant $sum_constant
  python base1_utils/base1_plot_extract_res.py --exp_root_nr "$extract_root" --save_folder "$record_folder" --attack "noise" --add_info "gaussian" --sum_constant $sum_constant
  python base1_utils/base1_plot_extract_res.py --exp_root_nr "$extract_root" --save_folder "$record_folder" --attack "noise" --add_info "rand" --sum_constant $sum_constant
  python base1_utils/base1_plot_extract_res.py --exp_root_nr "$extract_root" --save_folder "$record_folder" --attack "delete" --add_info "row" --sum_constant $sum_constant
  python base1_utils/base1_plot_extract_res.py --exp_root_nr "$extract_root" --save_folder "$record_folder" --attack "insert" --add_info "concatenate" --sum_constant $sum_constant

  res_nr="$classify_res_folder/$classify_res_file"
  fig_save_folder="$exp_root/$exp_tag/$dataset/res_figures"
  python base1_utils/base1_plot_together.py --res_nr "$res_nr" --record_folder "$record_folder" --save_folder "$fig_save_folder" --attack "noise" --add_info "uniform"
  python base1_utils/base1_plot_together.py --res_nr "$res_nr" --record_folder "$record_folder" --save_folder "$fig_save_folder" --attack "noise" --add_info "gaussian"
  python base1_utils/base1_plot_together.py --res_nr "$res_nr" --record_folder "$record_folder" --save_folder "$fig_save_folder" --attack "noise" --add_info "rand"
  python base1_utils/base1_plot_together.py --res_nr "$res_nr" --record_folder "$record_folder" --save_folder "$fig_save_folder" --attack "delete" --add_info "row"
  python base1_utils/base1_plot_together.py --res_nr "$res_nr" --record_folder "$record_folder" --save_folder "$fig_save_folder" --attack "insert" --add_info "concatenate"

fi
