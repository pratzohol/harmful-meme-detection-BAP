# hateful-meme-detection

## Resources:

1. [Official site of OSPC](https://ospc.aisingapore.org/)
2. [Submission Guide for OSPC](https://github.com/AISG-Technology-Team/AISG-Online-Safety-Challenge-Submission-Guide)

## Reference Repos

<table>
<tr>
<th colspan=3>Research Papers</th>
<th rowspan=2>Github Repo</th>
</tr>
<tr>
<th>Model Name</th>
<th>Description</th>
<th>Paper URL</th>
</tr>
<tr>
<td>NA</td>
<td>
1. Used VisualBERT.
</td>
<td>https://arxiv.org/abs/2012.12975</td>
<td>https://github.com/rizavelioglu/hateful_memes-hate_detectron/tree/main</td>
</tr>
<tr>
<td>MOMENTA</td>
<td>
1. Used CLIP.<br>
2. Use of online google vision APIs for OCR, object detection, attribute detection
</td>
<td>https://arxiv.org/pdf/2109.05184</td>
<td>https://github.com/LCS2-IIITD/MOMENTA</td>
</tr>
<tr>
<td>PromptHate</td>
<td>

1. Extracts image text using [`EasyOCR`](https://github.com/JaidedAI/EasyOCR) <a href=https://github.com/JaidedAI/EasyOCR><code>EasyOCR</code></a><br>
2. This is followed by in-painting to remove the text from the image using <a href=https://github.com/open-mmlab/mmediting><code>MMEditing</code></a>.<br>
3. Generates image caption using <code>ClipCap</code> (pre-trained model : works well for low-res img).<br> 
4. Then, it uses Google vision web-entity detection API and <code>FairFace</code>(pre-trained model : extract demographic information of the person from image)<br>

</td>
<td>https://arxiv.org/pdf/2302.04156</td>
<td>https://gitlab.com/bottle_shop/safe/prompthate</td>
</tr>
</table>

## Datasets:
[Datasets for OSPC AI Singapore](https://drive.google.com/drive/folders/1n-60QbFi1XJzyJ7RXuJ7PKflDr6_qJKS?usp=sharing) can be found here. It contains the following datasets:

1. [Facebook harmful meme detection challenge dataset](https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/)
2. [Total defence memes - singapore](https://arxiv.org/pdf/2305.17911.pdf)
3. Palash's Sir Dataset (RMMHS)

## Models:

1. For LLMs, a 7B parameter model takes 28GB of GPU memory. It can be reduced to 14GB using float16 precision and to 7GB using int8 precision.

2. 

