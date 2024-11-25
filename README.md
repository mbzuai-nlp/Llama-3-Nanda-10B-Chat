# Llama-3-Nanda-10B-Chat

<!-- Provide a quick summary of what the model is/does. -->

Llama-3-Nanda-10B-Chat (or Nanda for short) is a 10 billion parameter pre-trained and instruction-tuned bilingual large language model for both Hindi and English, 
trained on a dataset containing 65 billion Hindi tokens. The model is based on transformer-based decoder-only (LLaMA-3) architecture. It implements Rotary Position Embeddings (RoPE), enabling the model to extrapolate
to long sequence lengths, providing improved context handling and model precision.


## How to Get Started with the Model:

Below is sample code to use the model. Note that the model requires a custom model class, so users must
enable `trust_remote_code=True` while loading the model.
The code below is tested on `transformers==4.28.0`.


```python

# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "MBZUAI/Llama-3-Nanda-10B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

prompt_hindi = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def get_response(text,tokenizer=tokenizer,model=model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.95,
        temperature=0.2,
        max_length=500,
        min_length=30,
        repetition_penalty=1.3,
        do_sample=True
    )
    
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    
    response = response.split("Response:\n")[1]
    return response


ques= "मुझे यूएई के बारे में कुछ रोचक तथ्य बताएं?"
text = prompt_hindi.format_map({'Question':ques})
print(get_response(text))

```



**Response:**

> <span style="color:#778899;"> यूएई (संयुक्त अरब अमीरात) एक ऐसा देश है जो मध्य पूर्व और खाड़ी क्षेत्र का हिस्सा है। यहां आपको इसके बारे में जानने योग्य कई दिलचस्प बातें मिलेंगी:
>
> <span style="color:#778899;"> 1. **बहुसांस्कृतिक समाज**: UAE को अक्सर "विश्व की खिड़की" कहा जाता है, क्योंकि यह विभिन्न संस्कृतियों, परंपराओं और धर्मों से प्रभावित होता है। दुबई जैसे शहरों में, आप दुनिया भर के लोगों द्वारा लाए गए विविध खाद्य पदार्थों, भाषाओं और जीवनशैलियों का अनुभव कर सकते हैं।
>
> <span style="color:#778899;"> 2. **आर्थिक विकासकर्ता**: संयुक्त अरब अमीरात ने तेल उद्योग पर आधारित अपनी अर्थव्यवस्था विकसित करके महत्वपूर्ण आर्थिक प्रगति देखी है। आज, इसका वैश्विक स्तर पर पर्यटन उद्योग, रियल एस्टेट विकास, वित्तीय सेवाओं आदि क्षेत्रों में भी प्रमुख योगदान है।
>
> <span style="color:#778899;"> 3. **प्रौद्योगिकी नवाचार केंद्र**: अबू धाबी प्रौद्योगिकी हब बन रहा है। 2017 में, शहर ने घोषणा की कि वह $135 बिलियन खर्च करेगा ताकि अगले दशक तक अपने बुनियादी ढांचे को उन्नत किया जा सके, जिसमें स्मार्ट सिटी परियोजनाओं सहित अन्य परियोजनाएं शामिल होंगी।
>
> <span style="color:#778899;"> 4. **वास्तुकला चमत्कार**: बुर्ज खलीफा जैसी गगनचुंबी इमारतें, पाम जुमेराह द्वीप या अबु धाबी में शेख जायद मस्जिद - ये सभी वास्तुकला प्रतिभा के अद्भुत नमूने हैं जिन्होंने विश्व मानचित्र पर अपना निशान बनाया है।
>
> <span style="color:#778899;"> 5. **अमीर विरासत**: यूएई प्राचीन सभ्यताओं का घर था, जैसा कि अल-ऐन नामक स्थल पर, जहां पुरातत्वविदों ने लगभग 7000 ईसा पूर्व डेटिंग वाले मानव बस्तियों के अवशेष खोजे। इस प्रकार, संस्कृतिक धरोहर समृद्ध एवं विविधतापूर्ण है।
>
> <span style="color:#778899;"> 6. **खेलों में अग्रणी भूमिका**: खेल आयोजन आयोजित करने में यूएई हमेशा आगे रहता है। इसने फुटबॉल (सॉकर), टेनिस और फॉर्मूला वन रेसिंग समेत अनेक अंतरराष्ट्रीय प्रतियोगिताओं की मेजबानी की है।
>
> <span style="color:#778899;"> 7. **शिक्षा प्रणाली**: शिक्षा सरकार की प्राथमिकता रही है; हाल ही में पेश किए गए राष्ट्रीय पाठ्यक्रम सुधार कार्यक्रम (NCIP) का उद्देश्य छात्रों को भविष्य के लिए तैयार करना तथा उन्हें कौशल प्रदान करना, जिससे वे तेजी से बदलती हुई नौकरी बाजार में प्रतिस्पर्धात्मक रूप से काम कर सकें।
>
> <span style="color:#778899;"> ये सिर्फ कुछ उदाहरण हैं, लेकिन ऐसे बहुत सारे कारण मौजूद हैं जिनके चलते यूएई वास्तव में देखने लायक स्थान है!


### Model Details:

- **Developed by:** [Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)](https://mbzuai.ac.ae/), [Inception](https://www.inceptionai.ai/), and [Cerebras Systems](https://www.cerebras.net/)
- **Language(s) (NLP):** Hindi (and English)
- **License:** Llama 3
- **Input:** Text-only data
- **Output:** Model generates text
- **Paper :** [Llama-3-Nanda-10B-Chat: An Open Generative Large Language Model for Hindi](https://github.com/mbzuai-nlp/Llama-3-Nanda-10B-Chat/blob/main/Llama-3-Nanda-10B-Chat-Paper.pdf)


## Intended Use

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
We release Nanda under Meta’s Llama 3 license, and users must adhere to the terms and conditions of the
[license](https://www.llama.com/llama3/license/), Meta’s [acceptable use policy](https://www.llama.com/llama3/use-policy/), Meta’s [privacy policy](https://www.facebook.com/privacy/policy/), and the applicable policies, laws, and regu-
lations governing the specific use-case and region. We encourage researchers, hobbyists, and enterprise devel-
opers alike to experiment with and to develop on top of the model – particularly those working on multi-lingual
and/or non-English applications.

We welcome all feedback and opportunities to collaborate.

This model is a release from the MBZUAI-Inception-Cerebras parternship, and at the time of release, 
achieved state-of-the-art across a comprehensive Hindi test suite.
Some potential downstream uses include:

- *Research*: This model can be used by researchers and developers.
- *Commercial Use*: It can be used as a base model to further fine-tune for specific use cases.
Some potential use cases include:
  - Chat-assistants
  - Customer service
 
Audiences that we hope will benefit from our model:
- *Academics*: For those researching Hindi natural language processing.
- *Businesses*: Companies targeting Hindi-speaking audiences.
- *Developers*: Those integrating Hindi language capabilities in apps.


### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

While Llama-3-Nanda-10B-Chat is a powerful Hindi and English bilingual model, it's essential to understand its limitations and the potential of misuse. 
It is prohibited to use the model in any manner that violates applicable laws or regulations. 
The following are some example scenarios where the model should not be used.

- *Malicious Use*: The model should not be used for generating harmful, misleading, or inappropriate content. This includes but is not limited to:
   - Generating or promoting hate speech, violence, or discrimination
   - Spreading misinformation or fake news
   - Engaging in or promoting illegal activities

- *Sensitive Information*: The model should not be used to handle or generate personal, confidential, or sensitive information.

- *Generalization Across All Languages*: Llama-3-Nanda-10B-Chat is bilingual and optimized for Hindi and English, it should not be assumed to have equal proficiency in other languages.

- *High-Stakes Decisions*: The model should not be used to make high-stakes decisions without human oversight. This includes medical, legal, financial, or safety-critical decisions.




## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

<!-- The model is trained on publicly available data which was in part curated by Inception. -->
We have employed different techniqes to reduce bias in the model. While efforts have been made to minimize biases, it is likely that the model, as with all LLM models, will exhibit some bias. 

The model is trained as an AI assistant for Hindi and English speakers. The model is limited to produce responses for queries in these two languages
and may not produce appropriate responses to other language queries.

By using Llama-3-Nanda-10B-Chat, you acknowledge and accept that, as with any large language model, it may generate incorrect, misleading and/or offensive information or content. 
The information is not intended as advice and should not be relied upon in any way, nor are we responsible for any of the content or consequences resulting from its use. 
We are continuously working to develop models with greater capabilities, and as such, welcome any feedback on the model


## Training Details:

### Training Data:


<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

For the pre-training of Llama-3-Nanda-10B-Chat, we used a diverse bilingual corpus sourced from the Web and other sources. We also used publicly available English and code datasets.
To collect Hindi data, we used multiple sources including web pages, Wikipedia articles, news articles, Hindi books, etc.


### Training Procedure:

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

We performed continuous pre-training followed by instruction tuning, both on Cerebras supercomputer. 


## Evaluation:

<!-- This section describes the evaluation protocols and provides the results. -->

We conducted a comprehensive evaluation of Llama-3-Nanda-10B-Chat and benchmarked it against several other leading base language models, focusing on both English and Hindi. The evaluation criteria spanned various dimensions, including:

- **Knowledge:** How well the model answers factual questions.
- **Reasoning:** The model's ability to answer questions requiring reasoning.
- **Misinformation/Bias:** Assessment of the model's susceptibility to generating false or misleading information, and its neutrality.




Hindi-only evaluation results:


| **Model Name**                              | **Average**   | **MMLU-0shot** | **MMLU-5-shot** | **HellaSwag-0shot** | **ARC-Easy_0shot** | **ARC-Challenge-0-shot** | **TruthfulQA-0-shot** |
|---------------------------------------------|---------------|-----------------------------|------------------------------|----------------------------------|-----------------------------------|------------------------------------|-------------------------|
| Google/Gemma-2-9b-base                      | 30.2          | 26.9                        | 27.2                         | 27.1                             | 28.2                              | 23.6                               | 48.2                    |
| meta-llama/Llama-2-7b                       | 31.02         | 27.9                        | 28.1                         | 29.6                             | 29.3                              | 24.9                               | 46.3                    |
| meta-llama/Llama-2-13b                      | 31.3          | 28.3                        | 29.3                         | 30.6                             | 29.2                              | 26.6                               | 43.8                    |
| BhabhaAI/Gajendra-v0.1-7B                    | 31.47         | 27.4                        | 27.9                         | 33.0                             | 36.7                              | 26.6                               | 37.2                    |
| ai4bharat/Airavata-7B                          | 32.02         | 28.1                        | 28.5                         | 33.0                             | 32.0                              | 25.6                               | 44.9                    |
| sarvamai/sarvam-2b-v0.5                     | 37.7          | 28.3                        | 29.1                         | 46.2                             | 45.8                              | 32.3                               | 44.5                    |
| AryaBhatta-GemmaOrca-Merged-8.5B        | 39.43         | 31.4                        | 35.9                         | 42.6                             | 46.5                              | 32.7                               | 47.5                    |
| meta-llama/Meta-Llama-3-8b                  | 39.83         | 30.2                        | 37.3                         | 45.7                             | 45.9                              | 34.5                               | 45.4                    |
| CohereForAI/Aya-23-8B                       | 40.18         | 29.8                        | 36.8                         | 48.4                             | 48.3                              | 33.9                               | 43.9                    |
| meta-llama/Llama-3.1-8B                     | 40.42         | 29.9                        | 37.3                         | 46.9                             | 50.2                              | 34.3                               | 43.9                    |
| AryaBhatta-GemmaUltra-Merged-8.5B       | 41.18         | 34.6                        | 37.5                         | 45.5                             | 48.9                              | 33.4                               | 47.2                    |
| meta-llama/Llama-3.1-8B-Instruct            | 41.8          | 32.9                        | 38.9                         | 48.0                             | 50.5                              | 36.2                               | 44.3                    |
| **Llama-3-Nanda-10B-Chat**                  | **47.88**     | **38.6**                    | **44.3**                     | **56.4**                         | **59.6**                          | **40.3**                           | **48.1**                |





English-only evaluation results:

| **Model Name**                                    | **Average** | **MMLU_0_shot** | **HellaSwag_0_shot** | **ARC_0_shot** | **TruthfulQA_0_shot** |
|---------------------------------------------------|-------------|-------------------|------------------------|------------------|-------------------------|
| Google/Gemma-2-9b-base                            | 33.03       | 28.4              | 33.1                   | 24.2             | 46.4                    |
| sarvamai/sarvam-2b-v0.5                           | 42.83       | 29.4              | 61.7                   | 42.5             | 37.7                    |
| ai4bharat/Airavata-7B                                | 44.53       | 31.7              | 65.5                   | 40.1             | 40.8                    |
| meta-llama/Llama-2-7b                             | 46.00       | 31.10             | 72.90                  | 40.50            | 39.50                   |
| ai4bharat/Gajendra-v0.1-7B                           | 48.55       | 37.5              | 73.0                   | 43.0             | 40.7                    |
| CohereForAI/Aya-23-8B                             | 49.63       | 34.0              | 73.9                   | 45.2             | 45.4                    |
| meta-llama/Llama-2-13b                            | 51.20       | 36.90             | 77.70                  | 46.10            | 44.10                   |
| AryaBhatta-GemmaOrca-Merged-8.5B              | 53.03       | 40.4              | 72.4                   | 45.4             | 53.9                    |
| AryaBhatta-GemmaUltra-Merged-8.5B             | 53.65       | 42.5              | 74.1                   | 45.4             | 52.6                    |
| meta-llama/Meta-Llama-3-8b                        | 53.65       | 39.2              | 79.1                   | 52.3             | 44.0                    |
| meta-llama/Llama-3.1-8B                           | 54.33       | 39.7              | 78.9                   | 53.5             | 45.2                    |
| meta-llama/Llama-3.1-8B-Instruct                  | 57.53       | 41.8              | 79.3                   | 55.1             | 53.9                    |
| **Llama-3-Nanda-10B-Chat**                            | **59.45**       | **48.7**              | **79.2**                   | **53.7**             | **56.2**                    |


### Recommendations

It is recommended that users:
- Avoid using the model in sensitive domains without human oversight.
- Verify the accuracy of factual information provided by the model.
- Regularly evaluate the model to ensure it aligns with ethical guidelines.


## Terms of use

By accessing this model, you are agreeing to the LLama 3 terms and conditions of the [license](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE), [acceptable use policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3/USE_POLICY.md) and [Meta’s privacy policy](https://www.facebook.com/privacy/policy/)


