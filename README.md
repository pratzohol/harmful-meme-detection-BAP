# hateful-meme-detection

## Leaderboard

![image](https://github.com/user-attachments/assets/caebdd25-36b3-47f6-91fc-d58c90557768)

## Resources

1. [Official site of OSPC](https://ospc.aisingapore.org/)
2. [Submission Guide for OSPC](https://github.com/AISG-Technology-Team/AISG-Online-Safety-Challenge-Submission-Guide)

## Reference Repos

<table>
<tr>
<th colspan=3>Research Papers</th>
</tr>
<tr>
<th>Model Name</th>
<th>Description</th>
<th>Link to Paper and Git repo</th>
</tr>
<!----- Row 1 ----->
<tr>
<td>NA</td>
<td>

1. Used VisualBERT.
</td>
<td>

[Paper](https://arxiv.org/abs/2012.12975) | [Git Repo](https://github.com/rizavelioglu/hateful_memes-hate_detectron/tree/main)
</td>
</tr>
<!----- Row 2 ----->
<tr>
<td>MOMENTA</td>
<td>
<img src="https://github.com/pratzohol/harmful-meme-detection/blob/main/img/momenta.png" alt="momenta"> <br>
<blockquote>Use of online google vision APIs for OCR, object detection, attribute detection.</blockquote>
</td>
<td>

[Paper](https://arxiv.org/pdf/2109.05184) | [Git Repo](https://github.com/LCS2-IIITD/MOMENTA)
</td>
</tr>
<!----- Row 3 ----->
<tr>
<td>PromptHate</td>
<td>

1. Extracts image text using [`EasyOCR`](https://github.com/JaidedAI/EasyOCR)
2. This is followed by in-painting to remove the text from the image using [`MMEditing`](https://github.com/open-mmlab/mmediting)
3. Generates image caption using `ClipCap` (pre-trained model : works well for low-res img)
4. Then, it uses Google vision web-entity detection API and `FairFace`(pre-trained model : extract demographic information of the person from image)
5. Then, the image caption and image text are passed through `RoBERTa` model to get the final prediction using MLM prompting.
</td>
<td>

[Paper](https://arxiv.org/pdf/2302.04156) | [Git Repo](https://gitlab.com/bottle_shop/safe/prompthate)
</td>
</tr>
<!----- Row 4 ----->
<tr>
<td>Hate-CLIPper</td>
<td>


1. Image _i_ and text _t_ is passed through CLIP image and text encoders to obtain unimodal features $f_i$ and $f_t$.
2. To align the text and image feature space, $f_i$ and $f_t$ are passed through a trainable projection layer.
3. We then get $p_i$ and $p_t$ which have the same dimensionality of _n_.
4. Then, a _feature interaction matrix_(FIM) is computed by taking the outer product of $p_i$ and $p_t$, i.e., FIM = $p_i \otimes p_t$.
5. We can do 3 things now :
    - Concat : concat the $p_i$ and $p_t$ to get a vector of dimension $2n$
    - Cross-fusion : Flatten the FIM to get a vector of dimension $n^2$
    - Align-fusion : Take the diagonal of the FIM to get a vector of dimension $n$.
6. It is then passed through FFN to obtain final classification.

> Doesn't use additional input features like object bounding boxes, face detection and text attributes.
</td>
<td>

[Paper](https://arxiv.org/pdf/2210.05916) | [Git Repo](https://github.com/gokulkarthik/hateclipper)
</td>
</table>

## Datasets
All the     [Datasets for OSPC AI Singapore](https://drive.google.com/drive/folders/1n-60QbFi1XJzyJ7RXuJ7PKflDr6_qJKS?usp=sharing) can be found here. It contains the following datasets:

1. [Facebook harmful meme detection challenge dataset](https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/)
2. [Total defence memes - singapore](https://arxiv.org/pdf/2305.17911.pdf)
3. Palash's Sir Dataset (RMMHS)
4. [Propanganda Meme Dataset](https://aclanthology.org/2021.acl-long.516.pdf)

## Notes

1. Tesseract OCR (`tessdata_best`) : Takes around "1 hrs 30 mins" (2.9 it/s) for 1800 images. Quite slow !!!
2. Tesseract OCR (`tessdata`) : Takes around "1 hr 10 mins" (2.3 it/s). Faster than `tessdata_best`.
3. In above two cases, turbo-boost was on. Now, turning off the turbo-boost, ran the `tessdata_best` on 272 images. Using `multiprocessing.Pool(4)`, it took "8 mins 33 secs". Using simple for-loop, it takes ">20 mins". Using `multiprocessing.Pool(3)`, it took "9 mins 06 secs". 
4. Scaling the time taken above to 1800 images, using multiprocess.Pool(4), it would take around "1 hr" only.
5. CLIP can handle images of size 224x224 upto 336x336.

6. Using HateCLIPper, the (auroc, acc) obtained on fb-meme data validation-set are:

    - _run_4_easyocr_ : (0.729, 0.614)
    - _run-3-easyocr_ : (0.739, 0.634)
    - _run-2-easyocr_ : (0.743, 0.646)
    - _run-1-easyocr_ : (0.740, 0.632)
    - _run-10_ : (0.70, 0.656)
    - _run-9_ : (0.733, 0.634)
    - _run-8_ : (0.5, 0.5)
    - _run-7_ : (0.737, 0.642)
    - _run-6_ : (0.7408, 0.666)

7. Using HateCLIPper, the (auroc, acc) obtained on RMMHS data are:

    - _run_4_easyocr_ : (0.815, 0.68)
    - _run-3-easyocr_ : (0.843, 0.67)
    - _run-2-easyocr_ : (0.865, 0.75)
    - _run-1-easyocr_ : (0.87, 0.73)
    - _run-10_ : (0.888, 0.789)
    - _run-9_ : (0.854, 0.68)
    - _run-8_ : (0.5, 0.45)
    - _run-7_ : (0.847, 0.789)
    - _run-6_ : (0.872, 0.835)

8. So, based on above data and charts from wandb, I decided to go with `run-1-easyocr` (`run-9` was second best contender).
9. Running the above model on translated val-set of fb-meme data, the (auroc, acc) obtained was (0.7456, 0.632)
10. Running the above model on translated val-set (sampling randomly 500) of fb-meme data, the (auroc, acc) obtained was (0.761, 0.722)
