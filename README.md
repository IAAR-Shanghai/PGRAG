<h1 align="center">
    🍄 PG-RAG: Empowering Large Language Models to Set up a Knowledge Retrieval Indexer via Self-Learning
</h1>


## Introduction

PG-RAG proposes a pre-retrieval augmented generation method that introduces a _refinement_ step before the _indexing-retrieval-generation_ process, ensuring the accuracy of retrieved content from the outset. We leverage the self-learning capabilities of LLMs to transform documents into easily understandable and retrievable hierarchical indexes. This process naturally filters out noise and enhances information readability. By establishing connections between similar or complementary pieces of knowledge, we enable the retriever to function across multiple documents. During the knowledge retrieval phase, we use _pseudo-answers_ to assist the retriever in locating relevant information and perform walking in the matrices, thereby achieving accurate and rapid knowledge localization. Finally, we assemble the retrieved fact paths into a structured context, providing rich background information for LLMs to generate knowledge-grounded responses. 

<p align="center"><img src="./assets/pgr.jpg" alt="" width="80%"></p>

Additionally, we've made XinhuaHallucinations creation process accessible through our open-source pipeline, [UHGEval-dataset](https://github.com/IAAR-Shanghai/UHGEval-dataset). This enables researchers to craft customized datasets.

<details><summary>Supported models of PG-RAG framework</summary>

| Model Type | Loading Method               | Example Models                     | References                                                                                                             |
|------------|------------------------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `api`      | `requests`                   | OpenAI models | [OpenAI API](https://platform.openai.com/docs/introduction)|
| `local`    | HuggingFace `Sentence transformers`   | `BAAI/bge-large-zh-v1.5`             | [HuggingFace `Sentence transformers`](https://sbert.net/)                                                 |
| `remote`   | `requests`                   | Internal use only                  | Not available to the public                                                                                            |

</details>

<details><summary>A specific case of main topic and fact-checking items.</summary>

```json
{
  "Main Topic": "Announcement of the results of national medical device supervision sampling",
  "Fact-Checking Items": {
    "Date and Time": "2023-07-28",
    "Issuing Organization": "National Medical Products Administration",
    "Types of Products Sampled": "Dental low-voltage electric motors, various medical patches (far infrared therapy patches, magnetic therapy patches, acupoint magnetic therapy patches), among other five types",
    "Sampling Results": "A total of 12 batches (units) of products did not meet the standard requirements",
    "Specific Non-compliant Products and Issues": {
      "Dental Low-Voltage Electric Motor (1 unit)": "Produced by Guangdong Jingmei Medical Technology Co., Ltd., with issues related to leakage current and patient auxiliary current (under working temperature), and no-load speed not meeting the standard requirements.",
      "Vertical Pressure Steam Sterilizer (1 unit)": "Produced by Hefei Huatai Medical Equipment Co., Ltd., involving 'permissible limit values of accessible parts under normal conditions' and limit values under single fault condition (ground fault) not meeting the standard requirements.",
      "Electric Suction Device (1 unit)": "Produced by Suzhou Bein Technology Co., Ltd., involving 'network-powered, movable high negative pressure/high flow equipment' not meeting the standard requirements.",
      "Patch-type Medical Devices (far infrared therapy patch, magnetic therapy patch, acupoint magnetic therapy patch) 6 batches": "Produced by Jiujiang Gaoke Pharmaceutical Technology Co., Ltd., Zhengzhou Zhongyuan Fuli Industrial & Trade Co., Ltd., Ulanqab Qiao's Weiye Medical Device Co., Ltd., Hunan Dexi Medical Technology Co., Ltd., and Chongqing Zhengren Medical Device Co., Ltd., with issues involving detection of 'pharmaceutical ingredients that should not be detected according to supplementary testing methods.'",
      "Human Blood and Blood Component Plastic Bag Containers (blood bags) 3 batches": "Produced by Nanjing Sailjin Biomedical Co., Ltd., with issues involving non-compliance of the blood bag transfusion ports with the standards."
    }
  }
}
```

</details>

<details><summary>A specific case of the mind map generated through main topic and fact-checking items.</summary>

```json
{
  "Announcement of the results of national medical device supervision sampling": {
    "Announcement Date and Time": "July 28, 2023",
    "Sampled Items": [
      "Dental Low-Voltage Electric Motors",
      "Patch-Type Medical Devices (including Far Infrared Therapy Patches, Magnetic Therapy Patches, Acupoint Magnetic Therapy Patches)"
    ],
    "Total Number of Sampled Products": 12,
    "Number of Product Types Not Meeting Standard Requirements": 5,
    "Non-Compliant Medical Devices and Manufacturer Information": {
      "Dental Low-Voltage Electric Motor": {
        "Quantity": 1,
        "Manufacturer": "Guangdong Jingmei Medical Technology Co., Ltd.",
        "Issue Description": "Involving leakage current and patient auxiliary current, no-load speed not meeting the standard requirements"
      },
      "Vertical Pressure Steam Sterilizer": {
        "Quantity": 1,
        "Manufacturer": "Hefei Huatai Medical Equipment Co., Ltd.",
        "Issue Description": "Involving the allowable limit values of accessible parts under normal conditions and limit values under single fault condition (ground fault) not meeting the standard requirements"
      },
      "Electric Suction Device": {
        "Quantity": 1,
        "Manufacturer": "Suzhou Bein Technology Co., Ltd.",
        "Issue Description": "Involving network-powered, movable high negative pressure/high flow equipment not meeting the standard requirements"
      },
      "Patch-Type Medical Devices": {
        "Sub-Types": [
          "Far Infrared Therapy Patch",
          "Magnetic Therapy Patch",
          "Acupoint Magnetic Therapy Patch"
        ],
        "Batch Quantity": 6,
        "List of Manufacturers": [
          "Jiujiang Gaoke Pharmaceutical Technology Co., Ltd.",
          "Zhengzhou Zhongyuan Fuli Industrial & Trade Co., Ltd.",
          "Ulangab Qiao's Weiye Medical Device Co., Ltd.",
          "Hunan Dexi Medical Technology Co., Ltd.",
          "Chongqing Zhengren Medical Device Co., Ltd."
        ],
        "Issue Description": "Involving detection of pharmaceutical ingredients that should not be detected according to supplementary testing methods"
      },
      "Human Blood and Blood Component Plastic Bag Containers (Blood Bags)": {
        "Quantity": 3,
        "Manufacturer": "Nanjing Sailjin Biomedical Co., Ltd.",
        "Issue Description": "Involving blood bag transfusion ports not meeting the standard requirements"
      }
    }
  }
}

```

</details>

<details><summary>A specific case of fact path in PG.</summary>

"Announcement of the results of national medical device supervision sampling"> "Non-Compliant Medical Devices and Manufacturer Information"> "Dental Low-Voltage Electric Motor"> "Manufacturer"> "Guangdong Jingmei Medical Technology Co., Ltd."

</details>

<details><summary>Project structure</summary>

```bash
.
├── .github
├── .gitignore
├── CITATION.bib
├── LICENSE
├── README.md
├── assets                  # Static files like images used in documentation
├── data                    # Datasets (e.g.,  1-Document QA)
├── output                  # Stores the contexts of querys
├── requirements.txt
└── pgrag                   # Source code for the project
    ├── configs             # Scripts for initializing model loading parameters
    ├── data             # Intermediate result data  
        ├── raw_news             # The original documents required for the retrieval library building
        ├── pg_gen             # Data during the pseudo-graph construction process  
        └── context_recall            # Data during the pseudo-graph retrieval process
    ├── mindmap_generator.py          # Scripts for generating mind maps
    ├── pseudo_graph_constructor.py             # Scripts for pseudo-graph construction
    ├── seed_context_recall.py           # Scripts for seed contexts recall
    ├── sub_pseudo_graph_retriever.py           # Scripts for structured contexts recall
    ├── llm                 # Calling LLM methods
    └── prompts             # Prompt Engineering
```

</details>

## Installation

Before using PG-RAG

1. Ensure you have Python 3.9.0+
2. `pip install -r requirements.txt`
3. Prepare your model configuration file:
    - Use [`pgrag/configs/real_config.py`].
    - In this new file, specify configurations for the models you wish to evaluate, including details like the model's name, type, token, private key, etc.

## Pipline

1. Generate the mind maps for original texts using `pgrag/mindmap_generator.py`.
2. Build pseudo-graph through `pgrag/pseudo_graph_constructor.py`.
3. Use `pgrag/seed_context_recall.py` for seed topics recall.
4. Use `pgrag/sub_pseudo_graph_retriever.py` final context extensions and context generation.

## Results for Experiment

<p align="center"><img src="./assets/main_results.jpg" alt=""></p>
