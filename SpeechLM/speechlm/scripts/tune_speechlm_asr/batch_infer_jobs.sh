
data=/home/ziqzhang/dataset/LibriSpeech/hubert_release_iter2_layer9_kmeans

# for num in 10k 40k 400k 4m; do
#     model=/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/finetune_asr/base_speechlmp_text${num}_16gpu_2accum/ctc60k_from_400k_bz1.6m_lr1e-5/checkpoint_best.pt
#     bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model $data
#     ehco "============================$model===================================="
#     bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model $data
#     ehco "============================$model===================================="
# done

# for num in 0.01 0.5 1.0 10; do
#     model=/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/finetune_asr/base_speechlmp_ctc${num}_16gpu_2accum/ctc60k_from_400k_bz1.6m_lr1e-5/checkpoint_best.pt
#     bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model $data
#     ehco "============================$model===================================="
#     bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model $data
#     ehco "============================$model===================================="
# done
# for num in wotext woswap; do
#     model=/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/finetune_asr/base_speechlmp_${num}_32gpu_1accum/ctc60k_from_400k_bz1.6m_lr1e-5/checkpoint_best.pt
#     bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model $data
#     ehco "============================$model===================================="
#     bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model $data
#     ehco "============================$model===================================="
# done


# model=/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/finetune_asr/base_speechlmp_wotext_woswap_32gpu_1accum/ctc60k_from_400k_bz1.6m_lr1e-5/checkpoint_best.pt
# bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model $data
# echo "============================$model===================================="
# bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model $data
# echo "============================$model===================================="


# model=/home/ziqzhang/data/tri4b10k_decode_a0.06/exp/finetune_asr/base_speechlmp_32gpu_1accum/ctc60k_from_400k_bz1.6m_lr1e-5/checkpoint_best.pt
# bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model $data
# echo "============================$model===================================="
# bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model $data
# echo "============================$model===================================="

# model=/home/ziqzhang/data/speechlm_tri4b_new_a0.06/exp/finetune_asr/base_speechlmp_32gpu_1accum/ctc60k_from_400k_bz1.6m_lr1e-5/checkpoint_best.pt
# bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model $data
# echo "============================$model===================================="
# bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model $data
# echo "============================$model===================================="


model=/home/ziqzhang/data/speechlm_tri4b_renew_a0.06/exp/finetune_asr/base_speechlmp_32gpu_1accum/ctc60k_from_400k_bz1.6m_lr1e-5/checkpoint_best.pt
bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model $data
echo "============================$model===================================="
bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model $data
echo "============================$model===================================="
