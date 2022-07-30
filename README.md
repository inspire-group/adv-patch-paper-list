# A Paper List for Localized Adversarial Patch Research

## Changelog

- <u>07/2022</u>: wrote two blog posts for adversarial patches and certifiably robust image classification [post1](https://freedom-to-tinker.com/2022/07/12/toward-trustworthy-machine-learning-an-example-in-defending-against-adversarial-patch-attacks/">) [post2](https://freedom-to-tinker.com/2022/07/19/toward-trustworthy-machine-learning-an-example-in-defending-against-adversarial-patch-attacks-2/)

- <u>03/2022</u>: **released the [leaderboard](https://github.com/inspire-group/patch-defense-leaderboard) for certified defenses for image classification against adversarial patches!!** 

- <u>11/2021</u>: added explanations of different defense terminologies. added a few more recent papers.

- <u>08/2021:</u> released the paper list!

## What is the localized adversarial patch attack?

Different from classic adversarial examples that are configured to have a small L_p norm distance to the normal examples, a localized adversarial patch attacker can <u>arbitrarily modify</u> the pixel values <u>within a small region</u>.

The attack algorithm is similar to those for the classic L_p adversarial example attack. You define a loss function and then optimize your perturbation to attain the attack objective. The only difference is that now 1) you can only optimize over pixels within a small region, 2) but within that region, the pixel values can be arbitrary as long as they are valid pixels.

Example of localized adversarial patch attack (image from [Brown et al.](https://arxiv.org/abs/1712.09665)):

<img src="asset/patch-example.png" width="60%" alt="patch image example" align=center>

## What makes this attack interesting?

<u>It can be realized in the physical world!</u>

Since all perturbations are within a small region, we can print and attach the patch in our physical world. This type of attack imposes a real-world threat on ML systems!

Note:  not all existing physically-realizable attacks are in the category of patch attacks, but the localized patch attack is (one of) the simplest and the most popular physical attacks.

## About this paper list

### Focus

1. <u>Test-time</u> attacks/defenses (do not consider localized backdoor triggers)
2. <u>2D computer vision tasks</u> (e.g., image classification, object detection, image segmentation)
3. <u>Localized</u> attacks (do not consider other physical attacks that are more "global", e.g., some stop sign attacks which require changing the entire stop sign background)

### Organization

1. I first categorize the papers based on the task: <u>image classification vs object detection</u> (and semantic segmentation, and other tasks)
2. I next group papers for <u>attacks vs defenses.</u>
3. I tried to organize each group of papers <u>chronically</u>. I consider two timestamps: the time when the preprint is available (e.g., arXiv) and the time when a (published) paper was submitted for peer-review. 

I am actively developing this paper list (I haven't added notes for all papers). If you want to contribute to the paper list, add your paper, correct any of my comments, or share any of your suggestions, feel free to reach out :)

## Table of Contents

- [**Defense Terminology**](#defense-terminology)
  
  - [Empirically Robust Defenses vs Provably/certifiably Robust Defenses](#empirically-robust-defenses-vs-provablycertifiably-robust-defenses)
  - [Robust Prediction vs Attack Detection](#robust-prediction-vs-attack-detection)

- [**Image Classification**](#image-classification)
  
  - [Attacks](#attacks)
  - [Certified Defenses](#certified-defenses)
  - [Certified Robustness Leaderboard](https://docs.google.com/spreadsheets/d/1zDBg5AmpWq92c_MaSx6vq4FsOUnzu57i8aUuex2NT7Y/edit?usp=sharing)
  - [Empirical Defenses](#empirical-defenses)

- [**Object Detection (and Semantic Segmentation)**](#object-detection-and-semantic-segmentation)
  
  - [Attacks](#attacks-1)
  
  - [Certified Defenses](#certified-defenses-1)
  
  - [Empirical Defenses](#empirical-defenses-1)

## Defense Terminology

### **Empirically Robust Defenses vs Provably/certifiably Robust Defenses**

There are two categories of defenses that have different robustness guarantees.

To evaluate the robustness of an <u>empirical defense</u>, we use <u>concrete attack algorithms</u> to attack the defense. The evaluated robustness does not have formal security guarantee:  it might be compromised by smarter attackers in the future.

To evaluate a <u>certified defense</u>, we need to develop a <u>robustness certification procedure</u> to determine whether the defense has certifiable/provable robustness for a given input against a given threat model. We need to formally prove that the certification results will hold for any attack within the threat model, including ones that have full knowledge of the defense algorithm, setup, etc.

**Notes:**  

1. We use certification procedure to evaluate the robustness of certified defenses; the certification procedure is agnostic to attack algorithms (i.e., it holds for any attack algorithm). As a bonus, this agnostic property lifts the burden of designing sophisticated adaptive attack algorithms for robustness evaluation. This is also why certified defenses usually do not provide "attack code" or "adversarial images" in their source code. 

2. Strictly speaking, it is unfair to directly compare the performance of empirical defenses and certified defenses due to their different robustness notions. 
   
   1. Some empirical defenses might have (seemingly) high *empirical robustness*. However, those empirical defenses have <u>zero</u> *certified robust accuracy*, and their empirical robust accuracy might drop greatly given a smarter attacker. 
   
   2. On the other hand, the evaluated certified robustness is a *provable lower bound* on model performance against any empirical adaptive attack within the threat model.

### Robust Prediction vs Attack Detection

There are also two different defense objectives (robustness notions).

<u>Robust prediction</u> aims to always make correct decisions (e.g., correct classification label, correct bounding box detection), even in the presence of an attacker.

<u>Attack detection</u>, on the other hand, only aims to detect an attack. If the defense detects an attack, it issues an alert and abstains from making predictions; if no attack is detected, it performs normal predictions. We can think of this type of defense as adding a special token "ALERT" to the model output space. 

**Notes:**

1. Apparently, the robust prediction defense is harder than the attack detection defense. The detect-and-alert setup might be problematic when human fallback is unavailable.
2. Nevertheless, we are still interested in studying attack detection defense. There is [an interesting paper](https://arxiv.org/abs/2107.11630) discussing the connection between these two types of defense.
3. How do we evaluate the robustness of attack detection defense:
   1. for clean images: perform the defense, and only consider a correct prediction when the output matches the ground-truth
   2. for adversarial images: perform the defense, count a robust prediction if the prediction output is ALERT or matches the ground-truth

## Image Classification

### Attacks

#### [Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition](https://dl.acm.org/doi/10.1145/2976749.2978392)

CCS 2016

1. uses "adversarial glasses" to fool face recognition models

#### [Is Deep Learning Safe for Robot Vision? Adversarial Examples against the iCub Humanoid](https://arxiv.org/abs/1708.06939)

arXiv 1708; ICCV Workshop 2017

1. adversarial example attack against robot vision
2. discusses the concept of localized perturbations, which could mimic attaching a sticker in the real world

#### [Adversarial Patch](https://arxiv.org/abs/1712.09665)

arXiv 1712; NeurIPS workshop 2017

1. **The first paper** that explicitly introduces the concept of **adversarial patch** attacks
2. Demonstrate a universal **physical** world attack

#### [LaVAN: Localized and Visible Adversarial Noise](https://arxiv.org/abs/1801.02608)

arXiv 1801; ICML 2018

1. Seems to be a concurrent work (?) as "Adversarial Patch"
2. Digital domain attack

#### [Perceptual-Sensitive GAN for Generating Adversarial Patches](https://ojs.aaai.org//index.php/AAAI/article/view/3893)

AAAI 2019

1. generate imperceptible patches.

#### [Generate (non-software) Bugs to Fool Classifiers](https://arxiv.org/abs/1911.08644)

AAAI 2020

#### [PatchAttack: A Black-box Texture-based Attack with Reinforcement Learning](https://arxiv.org/abs/2004.05682)

arXiv 2004; ECCV 2020

1. a *black-box* attack via reinforcement learning

#### [Robust Physical-World Attacks on Face Recognition](https://arxiv.org/pdf/2011.13526.pdf)

arXiv 2011

#### [A Data Independent Approach to Generate Adversarial Patches](https://link.springer.com/article/10.1007/s00138-021-01194-6)

Sprinter 2021

1. data independent attack; attack via increasing the magnitude of feature values

#### [Enhancing Real-World Adversarial Patches with 3D Modeling Techniques](https://arxiv.org/abs/2102.05334)

arXiv 2102

1. use 3D modeling to enhance physical-world patch attack

#### [Meaningful Adversarial Stickers for Face Recognition in Physical World](https://arxiv.org/abs/2104.06728)

arXiv 2104

1. add stickers to face to fool face recognition system

#### [Improving Transferability of Adversarial Patches on Face Recognition with Generative Models](https://arxiv.org/abs/2106.15058)

arXiv 2106; CVPR 2021

1. focus on transferability 

#### [Inconspicuous Adversarial Patches for Fooling Image Recognition Systems on Mobile Devices](https://arxiv.org/abs/2106.15202)

arXiv 2106; an old version is available at [arXiv 2009](https://arxiv.org/abs/2009.09774) 

1. generate *small (inconspicuous)* and localized perturbations

#### [Patch Attack Invariance: How Sensitive are Patch Attacks to 3D Pose?](https://arxiv.org/abs/2108.07229)

arXiv 2108

1. consider physical-world patch attack in the 3-D space (images are taken from different angles)

#### [Robust Adversarial Attack Against Explainable Deep Classification Models Based on Adversarial Images With Different Patch Sizes and Perturbation Ratios](https://ieeexplore.ieee.org/document/9548896)

IEEE Access

1. use patch to attack classification models and explanation models

#### [Adversarial Token Attacks on Vision Transformers](https://arxiv.org/abs/2110.04337)

arXiv 2110

1. An analysis of perturbing part of tokens of ViT

#### [One Thing to Fool them All: Generating Interpretable, Universal, and Physically-Realizable Adversarial Features](https://arxiv.org/abs/2110.03605)

arXiv 2110

#### [Generative Dynamic Patch Attack](https://arxiv.org/abs/2111.04266)

BMVC 2021

#### [Adversarial Mask: Real-World Adversarial Attack Against Face Recognition Models](https://arxiv.org/abs/2111.10759)

arXiv 2111

1. physical world attack via wearing a weird mask

#### [TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems](https://arxiv.org/abs/2111.09999)

arXiv 2111

1. natural-looking patch attacks

#### [Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?](https://arxiv.org/abs/2203.08392)

ICLR 2022

1. not exactly a patch attack. use pixel patches to attack ViT

#### [Defensive Patches for Robust Recognition in the Physical World](https://arxiv.org/abs/2204.06213)

CVPR 2022

1. not exactly an attack. use patch for good purposes...

#### [Adversarial Robustness is Not Enough: Practical Limitations for Securing Facial Authentication](https://dl.acm.org/doi/abs/10.1145/3510548.3519369)

IWSPA 2022

1. empirically analyze the robustness of face authetication against physical-world attacks

#### [Surreptitious Adversarial Examples through Functioning QR Code](https://www.mdpi.com/2313-433X/8/5/122/htm)

1. combining QR code with adversarial patch

#### [Adversarial Sticker: A Stealthy Attack Method in the Physical World](https://ieeexplore.ieee.org/abstract/document/9779913)

TPAMI

#### [Adversarial Patch Attacks and Defences in Vision-based Task: A Survey](https://arxiv.org/pdf/2206.08304.pdf)

arXiv 2206

1. a survey; no experimental evaluation
2. (TODO: check if this paper list misses any paper listed in this survey)

#### 

[(go back to table of contents)](#table-of-contents)



### Certified Defenses

#### *<u>Check out this [leaderboard](https://github.com/inspire-group/patch-defense-leaderboard) for certified robustness against adversarial patches!</u>*

The leaderboard provides a summary of all papers in this section!

#### [Certified Defenses for Adversarial Patches](https://arxiv.org/abs/2003.06693)

ICLR 2020

**The first certified defense**. 

1. Show that previous two empirical defenses (DW and LGS) are broken against an adaptive attacker
2. Adapt IBP (Interval Bound Propagation) for certified defense
3. Evaluate robustness against different shapes
4. Very expensive; only works for CIFAR-10 and small models

#### [Clipped BagNet: Defending Against Sticker Attacks with Clipped Bag-of-features](https://ieeexplore.ieee.org/document/9283860)

IEEE S&P Workshop on Deep Learning Security 2020

1. **Certified defense**; clip BagNet features
2. Efficient

#### [(De)Randomized Smoothing for Certifiable Defense against Patch Attacks](https://arxiv.org/abs/2002.10733)

arXiv 2002, NeurIPS 2020

1. **Certified defense**; adapt ideas of randomized smoothing for $L_0$ adversary
2. Majority voting on predictions made from cropped pixel patches
3. Scale to ImageNet but expensive

#### [Minority Reports Defense: Defending Against Adversarial Patches](https://arxiv.org/abs/2004.13799)

arXiv 2004; ACNS workshop 2020

1. **Certified defense** for *detecting an attack*
2. Apply masks to the different locations of the input image and check inconsistency in masked predictions
3. Too expensive to scale to ImageNet (?)

#### [PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking](https://arxiv.org/abs/2005.10884)

arXiv 2005; USENIX Security 2021

1. **Certified defense** framework with two general principles: small receptive field to bound the number of corrupted features and secure aggregation for final robust prediction
2. BagNet for small receptive fields; robust masking for secure aggregation, which detects and masks malicious feature values
3. Subsumes several existing and follow-up papers

#### [Efficient Certified Defenses Against Patch Attacks on Image Classifiers](https://arxiv.org/abs/2102.04154)

Available on ICLR open review in 10/2020; ICLR 2021

1. **Certified defense**
2. BagNet to bound the number of corrupted features; Heaviside step function & majority voting for secure aggregation
3. Efficient, evaluate on different patch shapes

#### [Certified Robustness against Physically-realizable Patch Attack via Randomized Cropping](https://openreview.net/forum?id=vttv9ADGuWF)

Available on ICLR open review in 10/2020

1. **Certified defense**
2. Randomized image cropping + majority voting
3. only probabilistic certified robustness

#### [PatchGuard++: Efficient Provable Attack Detection against Adversarial Patches](https://arxiv.org/abs/2104.12609)

arXiv 2104; ICLR workshop 2021

1. **Certified defense** for *detecting an attack*
2. A hybrid of PatchGuard and Minority Report

#### [ScaleCert: Scalable Certified Defense against Adversarial Patches with Sparse Superficial Layers](https://arxiv.org/abs/2110.14120)

NeurIPS 2021

1. **certified defense** for *attack detection*. a fun paper using ideas from both minority reports and PatchGuard++
2. The basic idea is to apply (pixel) masks and check prediction consistency
3. it further uses superficial important neurons (the neurons that contribute significantly to the shallow feature map values) to prune unimportant regions so that the number of masks is reduced.

#### [PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier](https://arxiv.org/abs/2108.09135)

arXiv 2108; USENIX Security 2022

1. **Certified defense** that is compatible with any state-of-the-art image classifier
2. huge improvements in clean accuracy and certified robust accuracy (its clean accuracy is close to SOTA image classifier)

#### [Certified Patch Robustness via Smoothed Vision Transformers](https://arxiv.org/abs/2110.07719)

arXiv 2110; CVPR 2022

1. **Certified defense.** ViT + [De-randomized Smoothing](https://arxiv.org/abs/2002.10733)
2. Drop tokens that correspond to pixel masks to greatly improve efficiency. 

#### [Towards Practical Certifiable Patch Defense with Vision Transformer](https://arxiv.org/abs/2203.08519)

CVPR 2022

1. **Certified defense.** ViT + [De-randomized Smoothing](https://arxiv.org/abs/2002.10733)
2. A progressive training technique for smoothed ViT 
3. Isolated (instead of global) self-attention to improve defense efficiency

#### [Zero-Shot Certified Defense against Adversarial Patches with Vision Transformers](https://arxiv.org/abs/2111.10481)

arXiv 2111

1. **certified defense** for *attack detection*.
2. The idea is basically [Minority Report](https://arxiv.org/abs/2004.13799) with Vision Transformer
3. *The evaluation of clean accuracy seems problematic... (feel free to correct me if I am wrong)*
   1. *For a clean image, the authors consider the model prediction to be correct even when the defense believes there is an attack*

http://scis.scichina.com/en/2022/170306.pdf

#### *<u>Check out this [leaderboard](https://github.com/inspire-group/patch-defense-leaderboard) for certified robustness against adversarial patches!</u>*

[(go back to table of contents)](#table-of-contents)



### Empirical Defenses

#### [On Visible Adversarial Perturbations & Digital Watermarking](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w32/Hayes_On_Visible_Adversarial_CVPR_2018_paper.pdf)

CVPR workshop 2018

1. The **first empirical defense**. Use saliency map to detect and mask adversarial patches.

#### [Local Gradients Smoothing: Defense against Localized Adversarial Attacks](https://arxiv.org/abs/1807.01216)

arXiv 1807; WACV 2019

1. An **empirical defense**. Use pixel gradient to detect patch and smooth in the suspected regions.

#### [Ally patches for spoliation of adversarial patches](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0213-4)

Journal of Big Data

1. An **empirical defense**, make prediction on pixel patches, and do majority voting
2. This paper is somehow missed by almost all relevant papers in this field (probably due to its venue); it only has one self-citation. However, its idea is quite similar to some certified defenses that are published in 2019-2020

#### [Defending Against Physically Realizable Attacks on Image Classification](https://arxiv.org/abs/1909.09552)

arXiv 1909, ICLR 2020

1. **Empirical defense** via adversarial training
2. Interestingly show that adversarial training for patch attack does not hurt model clean accuracy 
3. Only works on small images

#### [SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems](https://arxiv.org/abs/1812.00292)

arXiv 1812; IEEE S&P Workshop on Deep Learning Security 2020

1. **Empirical defense** that leverages the *universality* of the attack (inapplicable to non-universal attacks)

#### [Detecting Patch Adversarial Attacks with Image Residuals](https://arxiv.org/abs/2002.12504)

arXiv 2002

1. **empirical defense**

#### [Adversarial Training against Location-Optimized Adversarial Patches](https://arxiv.org/abs/2005.02313)

arXiv 2005, ECCV workshop 2020

1. **empirical defense** via adversarial training (in which the patch location is being optimized)

#### [Vax-a-Net: Training-time Defence Against Adversarial Patch Attacks](https://arxiv.org/abs/2009.08194)

arXiv 2009; ACCV 2020

#### [Robustness Out of the Box: Compositional Representations Naturally Defend Against Black-Box Patch Attacks](https://arxiv.org/abs/2012.00558)

arXiv 2012

1. empirical defense; directly use CompNet to defend against *black-box* patch attack (evaluated with PatchAttack)

#### [Compositional Generative Networks and Robustness to Perceptible Image Changes](https://ieeexplore.ieee.org/abstract/document/9400221)

CISS 2021

1. An **empirical defense** against *black-box* patch attacks
2. A direct application of CompNet

#### [Detecting Localized Adversarial Examples: A Generic Approach using Critical Region Analysis](https://arxiv.org/pdf/2102.05241.pdf)

arXiv 2102

1. empirical defense for attack detection

#### [A Novel Lightweight Defense Method Against Adversarial Patches-Based Attacks on Automated Vehicle Make and Model Recognition Systems](https://link.springer.com/article/10.1007/s10922-021-09608-6)

Journal of Network and Systems Management

1. **empirical defense.** require the assumption of horizontal symmetry of the image. only applicable to a certain scenario.

#### [Real-time Detection of Practical Universal Adversarial Perturbations](https://arxiv.org/pdf/2105.07334.pdf)

arXiv 2105

1. An **empirical defense** that uses the magnitude and variance of the feature map values to detect an attack 
2. focus more on the universal attack (both localized patch and global perturbations)

#### [**Defending against Adversarial Patches with Robust Self-Attention**](http://www.google.com/url?q=http%3A%2F%2Fwww.gatsby.ucl.ac.uk%2F~balaji%2Fudl2021%2Faccepted-papers%2FUDL2021-paper-102.pdf&sa=D&sntz=1&usg=AFQjCNGJoUxi79GSFdIifBdsXf7lD4g2Kg)

ICML UDL workshop

1. **empirical defense**. detect and remove outliers in ViT

#### [Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attack](https://arxiv.org/abs/2108.05075)

arXiv 2108

1. empirical defense; use universality 

#### [Defending Against Universal Adversarial Patches by Clipping Feature Norms](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Defending_Against_Universal_Adversarial_Patches_by_Clipping_Feature_Norms_ICCV_2021_paper.pdf)

ICCV 2021

1. **empirical defense** via clipping feature norm.
2. Oddly, this paper does not cite Clipped BagNet

#### [Efficient Training Methods for Achieving Adversarial Robustness Against Sparse Attacks](https://iccv21-adv-workshop.github.io/short_paper/OFC_main_45.pdf)

ICCV workshop 2021

#### [Detecting Adversarial Patch Attacks through Global-local Consistency](https://dl.acm.org/doi/abs/10.1145/3475724.3483606)

MM workshop 2021

1. **empirical defense**

#### [ImageNet-Patch: A Dataset for Benchmarking Machine Learning Robustness against Adversarial Patches](https://arxiv.org/abs/2203.04412)

arXiv 2203

1. A dataset for adversarial patches
2. *Clarification from the authors:* the main purpose of the ImageNet-Patch dataset is to provide a fast benchmark of models against patch attacks, but not strictly related to *defenses* against adversarial patches, which is why they did not cite any adversarial patch defense papers.
3. (I ran some experiments; the transferability to architectures like ViT and ResMLP seemed low)

[(go back to table of contents)](#table-of-contents)



## Object Detection (and Semantic Segmentation)

### Attacks

#### [DPATCH: An Adversarial Patch Attack on Object Detectors](https://arxiv.org/abs/1806.02299)

arXiv 1806; AAAI workshop 2019

1. The **first (?) patch attack against object detector**

#### [Fooling automated surveillance cameras: adversarial patches to attack person detection](https://arxiv.org/abs/1904.08653)

arXiv 1904; CVPR workshop 2019

1. using a rigid board printed with adversarial perturbations to evade the detection of a person

#### [On Physical Adversarial Patches for Object Detection](https://arxiv.org/abs/1906.11897)

arXiv 1906

1. interestingly show that a physical-world patch in the background (far away from the victim objects) can have malicious effect

#### [Universal Physical Camouflage Attacks on Object Detectors](https://arxiv.org/pdf/1909.04326.pdf)

arXiv 1909

#### [Seeing isn't Believing: Towards More Robust Adversarial Attack Against Real World Object Detectors](https://dl.acm.org/doi/10.1145/3319535.3354259)

CCS 2019

#### [Adversarial T-shirt! Evading Person Detectors in A Physical World](https://arxiv.org/abs/1910.11099)

arXiv 1910; ECCV 2020

1. use a non-rigid T-shirt to evade person detection

#### [Making an Invisibility Cloak: Real World Adversarial Attacks on Object Detectors](https://arxiv.org/abs/1910.14667)

arXiv 1910; ECCV 2020

1. wear an ugly T-shirt to evade person detection

#### [APRICOT: A Dataset of Physical Adversarial Attacks on Object Detection](https://arxiv.org/abs/1912.08166)

arXiv 1912; ECCV 2020

1. a dataset with annotated patch locations.

#### [Adaptive Square Attack: Fooling Autonomous Cars With Adversarial Traffic Signs](https://ieeexplore.ieee.org/document/9165820)

IEEE IoT-J 2020

#### [Adversarial Patch Camouflage against Aerial Detection](https://arxiv.org/abs/2008.13671)

arXiv 2008

#### [Fast Local Attack: Generating Local Adversarial Examples for Object Detectors](https://arxiv.org/abs/2010.14291)

arXiv 2010; IJCNN 2020

#### [DPAttack: Diffused Patch Attacks against Universal Object Detection](https://arxiv.org/abs/2010.11679)

arXiv 2010; CIKM workshop

#### [Object Hider: Adversarial Patch Attack Against Object Detectors](https://arxiv.org/abs/2010.14974)

arXiv 2010

#### [Dynamic Adversarial Patch for Evading Object Detection Models](https://arxiv.org/abs/2010.13070)

arXiv 2010

#### [RPATTACK: Refined Patch Attack on General Object Detectors](https://arxiv.org/abs/2103.12469)

arXiv 2103; ICME 2021

#### [IPatch: A Remote Adversarial Patch](https://arxiv.org/pdf/2105.00113.pdf)

arXiv 2105

#### [Evaluating the Robustness of Semantic Segmentation for Autonomous Driving against Real-World Adversarial Patch Attacks](https://arxiv.org/abs/2108.06179)

arXiv 2108

#### [Physical Adversarial Attacks on an Aerial Imagery Object Detector](https://arxiv.org/abs/2108.11765)

arXiv 2108

#### [You Cannot Easily Catch Me: A Low-Detectable Adversarial Patch for Object Detectors](https://arxiv.org/abs/2109.15177)

arXiv 2109

1. attack against object detectors that can also evade attack-detection models.

#### [Naturalistic Physical Adversarial Patch for Object Detectors](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Naturalistic_Physical_Adversarial_Patch_for_Object_Detectors_ICCV_2021_paper.pdf)

ICCV 2021

1. an improved attack from adversarial T-shirt. The patch looks more natural (e.g., a dog)

#### [Legitimate Adversarial Patches: Evading Human Eyes and Detection Models in the Physical World](https://dl.acm.org/doi/abs/10.1145/3474085.3475653)

MM 2021

1. an improved attack from adversarial T-shirt. The patch looks more natural (e.g., an Ivysaur!)

#### [Adversarial Texture for Fooling Person Detectors in the Physical World](https://arxiv.org/abs/2203.03373)

CVPR 2022

1. consider cameras from different angles

#### [On the Feasibility and Generality of Patch-based Adversarial Attacks on Semantic Segmentation Problems](https://arxiv.org/abs/2205.10539)

arXiv 2205

#### [Physical Adversarial Attack on a Robotic Arm](https://cposkitt.github.io/files/publications/physical_adversarial_attack_ral22.pdf)

RA-L 2022

1. use adversarial patches to attack object detectors, in a robotic arm setting

#### [Feasibility of Inconspicuous GAN-generated Adversarial Patches against Object Detection](https://arxiv.org/abs/2207.07347)

arXiv 2207

#### [Physical Passive Patch Adversarial Attacks on Visual Odometry Systems](https://arxiv.org/abs/2207.05729)

arXiv 2207

1. use adversarial patches to attack ML-based Visual Odometry


[(go back to table of contents)](#table-of-contents)

### Certified Defenses

#### [DetectorGuard: Provably Securing Object Detectors against Localized Patch Hiding Attacks](https://arxiv.org/abs/2102.02956)

arXiv 2102; CCS 2021

1. The **first certified defense** for patch hiding attack
2. Adapt robust image classifiers for robust object detection
3. Provable robustness at a negligible cost of clean performance

#### [ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking](https://arxiv.org/abs/2202.01811)

arXiv 2202

1. use pixel masks to remove the adversarial patch in a certifiably robust manner.
2. a significant improvement in certified robustness
3. also discuss different robustness notions

[(go back to table of contents)](#table-of-contents)

### Empirical Defenses

#### [Role of Spatial Context in Adversarial Robustness for Object Detection](https://arxiv.org/abs/1910.00068)

arXiv 1910; CVPR workshop 2020

1. The **first empirical defense**, adding a regularization loss to constrain the use of spatial information
2. only experiment on YOLOv2 and small datasets like PASCAL VOC

#### [Meta Adversarial Training against Universal Patches](https://arxiv.org/pdf/2101.11453.pdf)

arXiv 2101; ICML 2021 workshop

#### [Adversarial YOLO: Defense Human Detection Patch Attacks via Detecting Adversarial Patches](https://arxiv.org/abs/2103.08860)

arXiv 2103

1. **Empirical defense** via adding adversarial patches and a "patch" class during the training

#### [We Can Always Catch You: Detecting Adversarial Patched Objects WITH or WITHOUT Signature](https://arxiv.org/abs/2106.05261)

arXiv 2106

1. Two **empirical defenses** for patch hiding attack
2. Feed small image region to the detector; grows the region with some heuristics; detect an attack when YOLO detects objects in a smaller region but misses objects in a larger expanded region.

#### [Adversarial Pixel Masking: A Defense against Physical Attacks for Pre-trained Object Detectors](https://dl.acm.org/doi/abs/10.1145/3474085.3475338)

MM 2021

1. **Empirical defense.** Adversarially train a "MaskNet" to detect and mask the patch

#### [Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection](https://arxiv.org/abs/2112.04532)

CVPR 2022

#### [Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis](https://arxiv.org/abs/2203.07341)

arXiv 2203

#### [Defending Against Person Hiding Adversarial Patch Attack with a Universal White Frame](https://arxiv.org/abs/2204.13004)

#### [PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch](https://arxiv.org/abs/2207.01795)

arxiv 2207

1. detect adversarial patches and remove them based on their different textures. consider different tasks like image classification, object detection, and video analysis



[(go back to table of contents)](#table-of-contents)