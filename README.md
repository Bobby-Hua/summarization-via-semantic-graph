# Long Dialogue Summarization with AMR

Code for the paper: [Improving Long Dialogue Summarization with Semantic Graph Representation](https://preview.aclanthology.org/acl-23-ingestion/2023.findings-acl.871.pdf) 

Yilun Hua, Zhaoyuan Deng, and Kathleen McKeown

# Background

Long dialogues in meetings and TV shows present great challenges for abstractive summarization. They exhibit complex structures with varied speaker interactions and their key information is scattered across the text. Their datasets are often small, leading to model overfitting. 

Our work proposes a novel application of Abstract Meaning Representation (AMR) to address these issues, offering reliable semantic cues and structural information to reduce overfitting and improve model performance. 

# Code

Our code uses several repos from existing research on AMR and text summarization, including but not limited to [wl-coref](https://github.com/vdobrovolskii/wl-coref), [gtos](https://github.com/jcyk/gtos), and [dialogLM](https://github.com/microsoft/DialogLM). All references can be found in our paper’s References section. 
