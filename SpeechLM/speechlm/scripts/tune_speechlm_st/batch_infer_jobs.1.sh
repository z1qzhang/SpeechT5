
dist="/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/finetune_covost/base_speechlmp_32gpu_1accum"
set -e

for seed in 1; do
    for lang in ar ca de tr; do
        data=/LocalData/dataset/CommonVoice/v4/en/en-${lang}
        model=${dist}/legacy_en${lang}_from_400k_bz3.2m_lr1e-4_seed${seed}/checkpoint_best.pt
        bash speechlm/scripts/tune_speechlm_st/inference_base.sh $model $data ${lang} test
    done
done
