import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class RSReferSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        rs_refer_seg_data="rrsisd",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "rs_refer_seg")
        self.refer_seg_ds_list = rs_refer_seg_data.split(
            "||"
        )  # ['rrsisd', 'risbench']
        self.rs_refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == 'risbench':
                refer_seg_ds = {}
                refs_train = []
                refer_seg_ds["images"] = []
                refer_seg_ds['annotations'] = {}
                refer_txt = os.path.join(DATA_DIR,ds,"output_phrase_train.txt")
                with open(refer_txt, "r") as refers:
                    for refer in refers:
                        # example: train_14700_0.png The bridge extending over a body of water is a central connector between two areas.
                        img_pth = refer.split(' ')[0]
                        ref = refer.split('.png ')[1]
                        assert len(img_pth) > 0, 'image path is none in '+refer
                        assert len(ref) > 0, 'refer text is none in '+refer
                        image_id = img_pth.split('.')[0]
                        item = {
                            'file_name': os.path.join(DATA_DIR,ds,"img_rgb",img_pth),
                            'height': 512,
                            'width': 512,
                            'id': image_id
                        }
                        refer_seg_ds["images"].append(item)
                        ann = {
                            'image_id': image_id,
                            'segmentation': os.path.join(DATA_DIR,ds,"mask",img_pth),
                        }
                        refer_seg_ds['annotations'][image_id] = ann
                        refs_train.append({
                            'sentences': [{'sent': ref}],
                            'image_id': image_id,
                            'ann_id': image_id,
                            'ref_id': image_id
                        })

                print(
                    "dataset {} (train split) has {} images and {} annotations.".format(
                        ds,
                        len(refer_seg_ds["images"]),
                        len(refer_seg_ds["annotations"]),
                    )
                )

            else:
                splitBy = "unc"
                refer_api = REFER(DATA_DIR, ds, splitBy)


                ref_ids_train = refer_api.getRefIds(split="train")
                images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
                refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

                refer_seg_ds = {}
                refer_seg_ds["images"] = []
                loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

                for item in loaded_images:
                    item = item.copy()
                    if ds == 'rrsisd':
                        item["file_name"] = os.path.join(
                            DATA_DIR, "rrsisd/images/rrsisd/JPEGImages", item["file_name"]
                        )
                    elif ds == 'rs_reason':
                        item["file_name"] = os.path.join(
                            DATA_DIR, "rs_reason/images/rrsisd/JPEGImages", item["file_name"]
                        )
                    refer_seg_ds["images"].append(item)
                refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

                print(
                    "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                        ds,
                        splitBy,
                        len(refer_seg_ds["images"]),
                        len(refer_seg_ds["annotations"]),
                    )
                )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.rs_refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.rs_refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]
                        # risbench‘s mask is a single channel image, whose ann is a string
                        if type(ann["segmentation"]) == str:
                            m = cv2.imread(ann["segmentation"], cv2.IMREAD_GRAYSCALE)
                            m = cv2.resize(m, (image_info["width"], image_info["height"]))
                            m = m.astype(np.uint8)
                            
                        elif len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            # just the mask matrix
            if type(ann["segmentation"]) == str:
                m = cv2.imread(ann["segmentation"], cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (image_info["width"], image_info["height"]))
                m = (m == 255).astype(np.uint8)
                masks.append(m)
                continue

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)



        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
