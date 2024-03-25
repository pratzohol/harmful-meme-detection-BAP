# hateful-meme-detection

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

![momenta-arch](/img/momenta.png)

Use of online google vision APIs for OCR, object detection, attribute detection.
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
6. It is then passed through FFN to obtai final classification.

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

## Models

1. For LLMs, a 7B parameter model takes 28GB of GPU memory. It can be reduced to 14GB using float16 precision and to 7GB using int8 precision.
2. 

