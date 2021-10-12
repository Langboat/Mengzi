## Chinese multimodal pre-training model Mengzi-Oscar 
Mengzi-Oscar is trained based on the English multimodal pre-training model [Oscar](https://github.com/microsoft/Oscar)，initialized with [Mengzi-Bert-base](https://huggingface.co/Langboat/mengzi-bert-base), 
using 3.7M image-text pairs, including 0.7M Chinese Image-Caption pairs and 3M Chinese Image-Question pairs, for a total of 0.22M images.
##  Mengzi-Oscar - Download
**Pre-training Model Download：**  [Mengzi-Oscar](https://huggingface.co/Langboat/mengzi-oscar-base)。  
**Downstream Task Model Download：**  [Chinese Image Caption](https://huggingface.co/Langboat/mengzi-oscar-base-caption).  [Chinese Image-Text Retrieval](https://huggingface.co/Langboat/mengzi-oscar-base-retrieval).
## Chinese Image Caption Demo（Randomly select from the AIC-ICC val set）
![image](https://github.com/ckmstydy/Mengzi/blob/main/Demo_images/1.png)  
**Generated Chinese Caption：绿油油的草地上有两个面带微笑的人在骑马。**   
**English Version (translated for reference)：two smiling men are riding horses on the green grass.**     

<br>

![image](https://github.com/ckmstydy/Mengzi/blob/main/Demo_images/2.png)  
**Generated Chinese Caption：两个打着伞的人和一个背着孩子的男人走在被水淹没的道路上。**  
**English Version (translated for reference)：Two people with umbrellas and a man with a child on his back walked along the flooded road.**     

## Quick Start（Pre-training / Image Caption / Retrieval）
### Installation -- Install [Oscar](https://github.com/microsoft/Oscar) via github
Check [INSTALL.md](https://github.com/microsoft/Oscar/blob/master/INSTALL.md) for installation instructions.

### Pre-training
#### 1）Data preparation for Pre-training
Mengzi-Oscar used 3.7M Chinese Image-text pairs with the following data source distribution:
| Source | VQA<br>(train) | GQA<br>(bal-train) | VG-QA<br>(train)	| COCO<br>(train) | Flicker30k<br>(train)|
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Image/Text | 83k/545k | 79k/1026k | 87k/931k | 112k/559k | 29k/145k |

Image objects detection, feature extraction:
We use the open source project X152-C4 object-attribute detection as a object detection tool , the project address: [Scene Graph Benchmark Repo](https://github.com/microsoft/scene_graph_benchmark).  
Pre-trained X152-C4 model [download address](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth)。 
Features are extracted by the following command:
```
# pretrained models at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
# the associated labelmap at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 \
MODEL.WEIGHT <path of vinvl_vg_x152c4.pth> \
MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
DATA_DIR <path of image feature> \
OUTPUT_DIR <path to save extracted features> \
TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True TEST.OUTPUT_FEATURE True
```
For the English label results of object detection, we provide [en-to-zh word dictionary](https://github.com/ckmstydy/Mengzi/blob/main/chinese_label.json),
you can convert English labels to Chinese labels by it. The pre-training data format, downstream task data format, and the original English data are visible in the open source project [Oscar VinVL_DOWNLOAD.md](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md).

#### 2）Run pre-training commands（based on Mengzi bert base）
```
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_oscarplus_pretrain.py \
--use_b 1 --max_grad_norm 10.0 \
--gradient_accumulation_steps 1 --use_img_layernorm 1 \
--output_dir <output floder to save the pretrained model> \
--bert_model bert --do_lower_case \
--model_name_or_path <path of mengzi bert base model> \
--learning_rate 1e-04 --warmup_steps 0 --do_train --max_seq_length 35 \
--on_memory --max_img_seq_length 50 --img_feature_dim 2054 --drop_out 0.1 \
--train_batch_size 1024 --ckpt_period 10000 --max_iters 2000000 --log_period 1000 \
--data_dir <path of pretraining data> \
--dataset_file coco_flickr30k_gqa.yaml \
--textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 8 
```
### Chinese Image Caption（fine-tune on COCO & AIC-ICC）
#### 1）data preparation for fine-tuning
See the object detection and feature extraction methods of pre-training data.
#### 2）fine-tune
fine-tune on COCO image caption dataset（8 RTX 3090 24G）
```
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_captioning.py \
--data_dir < path of downloaded coco dataset > \
--model_name_or_path <pat of pretrained Mengzi-Oscar model> \
--do_train --do_lower_case --add_od_labels --learning_rate 3e-5 \
--per_gpu_train_batch_size 128 --num_train_epochs 60 --tie_weights --freeze_embedding \
--label_smoothing 0.1 --drop_worst_ratio 0.2 --drop_worst_after 20000 \
--output_dir <path to save the fine-tune model> --num_workers 8
```
  
fine-tune on AIC-ICC train set, and inference on validation set（8 RTX 3090 24G）  
```
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_captioning.py \
--data_dir < path of AIC-ICC dataset > \
--model_name_or_path <path of pretrained model or finetuned coco caption model> \
--do_train --do_lower_case --add_od_labels --learning_rate 3e-5 \
--per_gpu_train_batch_size 128 --num_train_epochs 60 --tie_weights --freeze_embedding \
--label_smoothing 0.1 --drop_worst_ratio 0.2 --drop_worst_after 20000 \
--output_dir <path to save the finetuned model> --save_steps 1000 --logging_steps 1000 \
--evaluate_during_training --num_workers 8 --num_beams 5
```

inference on dataset
```
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_captioning.py \
--data_dir <path of test dataset> \
--do_test --test_yaml test_ch.yaml \
--num_beams 5 --per_gpu_eval_batch_size 128 --max_gen_length 20 \
--eval_model_dir <path of fine-tuned Chinese Image Caption model>
```

### Chinese Image-Text Retrieval（fine-tune on COCO and inference on AIC-ICC）
We fine-tune the pre-trainig model on COCO_ir dataset, and randomly select 1K pictures from the AIC-ICC validation set (each picture contains 5 ground truth captions) for evaluation.
#### 1）data preparation for fine-tuning  
See the object detection and feature extraction methods of pre-training data.
#### 2）fine-tune
fine-tune on COCO_ir dataset:
```
python oscar/run_retrieval.py --model_name_or_path <path of pretrained model>\
--data_dir <path of coco_ir> \
--img_feat_file <path of pretraining coco features.tsv>\
--do_train --do_lower_case --evaluate_during_training --num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt --per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 30 --weight_decay 0.05  \
--add_od_labels --od_label_type vg --max_seq_length 70 --max_img_seq_length 70 \
--output_dir <path to save mdoel> --save_steps 5000 --logging_steps 500
```
evaluation on the AIC-ICC validation 1k dataset:
```
python mengzi-oscar/run_retrieval.py --do_test --do_eval --test_split val \
--num_captions_per_img_val 5 --cross_image_eval --per_gpu_eval_batch_size 1024 \
--eval_model_dir <path of fintune model> --do_lower_case --add_od_labels \
--num_workers 4 --img_feat_file < path of AIC-ICC val.feature.pt > \
--data_dir <path of AIC-ICC-ir> --eval_img_keys_file val_img_keys_1k.pt
```
