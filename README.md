# SMN-ATTACK: SALIENT REGION-AWARE MULTI-SCALE NEIGHBORHOOD ATTACK FOR TRANSFERABLE VISION-LANGUAGE MODELS
## Requirements
See in `requirements.txt`.

### Prepare datasets and models
Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in`./configs/Retrieval_flickr.yaml, image_root`.
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).
Download the DINO model weights,[ DINO model](https://huggingface.co/facebook/dino-vits16/tree/main)

## Attack evaluation

From ALBEF to others models on the Flickr30k dataset:
```
python eval.py --config ./configs/Retrieval_flickr.yaml \
    --cuda_id 0 \
    --model_list ALBEF TCL CLIP_ViT CLIP_CNN \
    --source_model CLIP_CNN \
    --albef_ckpt ./checkpoints/albef_flickr.pth \
    --tcl_ckpt ./checkpoints/tcl_flickr.pth \
    --original_rank_index_path ./std_eval_idx/flickr30k/
```
