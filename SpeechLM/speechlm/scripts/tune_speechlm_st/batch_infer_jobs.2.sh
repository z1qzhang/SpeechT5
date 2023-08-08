
dist="/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/finetune_covost/base_speechlmp_32gpu_1accum"
set -e
export CUDA_VISIBLE_DEVICES=0

# for seed in 2; do
#     for lang in ca de tr; do
#         data=/LocalData/dataset/CommonVoice/v4/en/en-${lang}
#         model=${dist}/legacy_en${lang}_from_400k_bz3.2m_lr1e-4_seed${seed}/checkpoint_best.pt
#         bash speechlm/scripts/tune_speechlm_st/inference_base.sh $model $data ${lang} test
#     done
# done

for seed in 3; do
    for lang in ar; do  #ca de tr
        data=/LocalData/dataset/CommonVoice/v4/en/en-${lang}
        model=${dist}/legacy_en${lang}_from_400k_bz3.2m_lr1e-4_seed${seed}/checkpoint_best.pt
        bash speechlm/scripts/tune_speechlm_st/inference_base.sh $model $data ${lang} test
    done
done
