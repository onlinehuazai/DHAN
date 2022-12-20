# DHAN

The whole project description will be improved recently.


# Cite
@article{FENG2023109935,

    title = {DHAN: Encrypted JPEG image retrieval via DCT histograms-based attention networks},

    journal = {Applied Soft Computing},

    volume = {133},

    pages = {109935},

    year = {2023},

    issn = {1568-4946},

    doi = { https://doi.org/10.1016/j.asoc.2022.109935 },

    url = {https://www.sciencedirect.com/science/article/pii/S156849462200984X},

    author = {Qihua Feng and Peiya Li and Zhixun Lu and Zhibo Zhou and Yongdong Wu and Jian Weng and Feiran Huang},

    keywords = {Image retrieval, Image encryption, Attention networks, JPEG, Deep learning, Privacy-preserving},

    abstract = {In image retrieval, the images to be retrieved are stored on remote servers. Since the images contain amounts of privacy information and the server     cannot be fully trusted, people usually encrypt their images before uploading them to the server, which raises the demand for encrypted image retrieval (EIR). Current EIR techniques extract ruled hand-craft features from cipher images first and then build retrieval models (e.g., support vector machine, SVM) by these features, or use deep learning models (e.g., Convolutional Neural Network, CNN) to learn cipher-image representations in an end-to-end manner. However, SVM is not skilled at learning non-linear embedding in complex image database, and end-to-end EIR leads to low image security or retrieval performance because CNN is sensitive to extreme chaotic cipher images. Not-end-to-end EIR offers excellent encryption performance, and deep learning can further mine non-linear embedding from ruled hand-craft features. To this end, we propose a novel EIR scheme, named discrete cosine transform (DCT) Histograms-based Attention Networks (DHAN), which is based on deep learning to enhance expression ability of cipher-image in a not-end-to-end manner. Specifically, the DCT coefficients of images are encrypted by value replacement and block permutation encryption, and then the effective histogram features of DCT coefficients are extracted from the cipher images since the sets of DCT frequency in encrypted images are similar to that of plain images. After that, to dynamically learn the salient features of cipher images, we propose a new module named ResAttention and design deep attention networks to provide retrieval. Extensive experiments on two datasets demonstrate that DHAN not only provides high image security but also greatly improves retrieval performance than that of existing schemes.}
}
