# **Hugging Face Model Comparison**

The goal of this repository is to experiment with open source models using Hugging Face `transformers` in Python, and to compare how different models behave on the same task. I wish to document what I learn about each task along the way, including where the models work well and where they break.

There are 3 notebooks, each on a different task. All of them rely on the `pipeline` API from Hugging Face Transformers.
1) Sentiment Analysis
2) Speech-to-Text
3) Text Generation

## **Table of Contents**
- [Notebooks](#notebooks)
- [Key Libraries & Functions](#key-libraries--functions)
- [Workings](#workings)
- [Setup](#setup)

## **Notebooks**

| Stage | Idea | Notebook |
| :--- | :--- | :--- |
| **Sentiment Analysis** | test a pretrained sentiment classifier on airline tweets | [View](./Sentiment_analysis_models.ipynb) |
| **Speech-to-Text** | compare five speech recognition models on one audio clip | [View](./Speech%20to%20text%20models%20Comparision.ipynb) |
| **Text Generation** | continue Robert Frost lines and other prompts with a text generation model | [View](./text_generation.ipynb) |

## **Key Libraries & Functions**
Here are the important libraries used in my projects:

### **Hugging Face Transformers**
This was used in all three notebooks. Documentation: [Transformers](https://huggingface.co/docs/transformers/index), [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines).

| Function / Class | Role |
| :--- | :--- |
| transformers.pipeline("sentiment-analysis") | Builds a [text classification pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline) that returns a label and score per text. |
| transformers.pipeline("text-generation") | Builds a [text generation pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline) that continues a prompt. |
| transformers.pipeline("automatic-speech-recognition", model=...) | Builds an [ASR pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) that returns the transcription as `"text"`. |
| gen(prompt, max_length=..., num_return_sequences=...) | Generates text. `max_length` caps the length, `num_return_sequences` returns several continuations. |

### **PyTorch**
Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html). It is the backend that runs the models inside the pipelines. It is imported directly in the sentiment and speech notebooks, but the pipelines are what call it.

### **scikit-learn, pandas, NumPy, seaborn, and Matplotlib**
These were used for evaluation and visualization in the sentiment notebook, and pandas was also used to display the speech comparison table. Documentation: [scikit-learn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/), [seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/stable/index.html).

| Function | Role |
| :--- | :--- |
| [sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="true")](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) | Builds the confusion matrix, normalized per actual class. |
| [pandas.read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) / [pandas.DataFrame()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) | Loads the CSVs and builds the comparison table. |
| [seaborn.countplot()](https://seaborn.pydata.org/generated/seaborn.countplot.html) | Plots tweet counts per sentiment class. |
| [seaborn.heatmap()](https://seaborn.pydata.org/generated/seaborn.heatmap.html) | Draws the confusion matrix. |
| numpy.mean() | Computes accuracy. |

### **Audio libraries**
Used in the speech-to-text notebook. Documentation: [librosa](https://librosa.org/doc/latest/index.html), [SoundFile](https://python-soundfile.readthedocs.io/), [IPython.display](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html).

| Function | Role |
| :--- | :--- |
| [librosa.load(path, sr=16000)](https://librosa.org/doc/latest/generated/librosa.load.html) | Loads audio resampled to 16 kHz. |
| IPython.display.Audio() | Plays the clip in the notebook. |

### **Data download and standard library**
Documentation: [kagglehub](https://github.com/Kaggle/kagglehub), [textwrap](https://docs.python.org/3/library/textwrap.html), [glob](https://docs.python.org/3/library/glob.html), [os](https://docs.python.org/3/library/os.html), [shutil](https://docs.python.org/3/library/shutil.html).

| Function / Module | Role |
| :--- | :--- |
| kagglehub.dataset_download() | Downloads the Kaggle datasets. |
| os.listdir(), shutil.copy() | Finds the CSV and copies it to `Tweets.csv`. |
| textwrap.fill() | Wraps generated text for printing. |
| glob.glob() | Finds the uploaded audio file. |

## **Workings**

### **Sentiment Analysis**

**Notebook:** [Sentiment_analysis_models.ipynb](./Sentiment_analysis_models.ipynb)

**Model:** `pipeline('sentiment-analysis')` with no model named, which defaulted to [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).

**Dataset:** [US Airline Sentiment tweets](https://www.kaggle.com/datasets/tango911/airline-sentiment-tweets) from Kaggle, downloaded with `kagglehub`.

**Hand-written tests:**
- Simple positive and negative sentences were classified correctly.
- Gibberish still received a label, one NEGATIVE (0.97) and one POSITIVE (0.77).
- Slang failed: "This is the shit" and "That is bad ass" were both labelled NEGATIVE.
- Clear words like "terrific" and "garbage" were classified correctly.

**Evaluation:** neutral tweets were dropped, leaving **2,659** positive and negative tweets. Each tweet was run through the pipeline and compared with its label.

| Metric | Result |
| :--- | :--- |
| **Accuracy** | **88.64 %** |
| Negative tweets predicted Negative | 0.8946 |
| Positive tweets predicted Positive | 0.8533 |

**Takeaway:** the model works well on standard text but struggles with slang and meaningless input, so informal social media text would likely need fine-tuning.

### **Speech-to-Text**

**Notebook:** [Speech to text models Comparision.ipynb](./Speech%20to%20text%20models%20Comparision.ipynb)

Five models were run through the same ASR pipeline on one audio clip, loaded at 16 kHz.

| Model | Transcription |
| :--- | :--- |
| [OpenAI Whisper Base](https://huggingface.co/openai/whisper-base) | Hi. Hello. One, two, three. I'm Zon Wake. Pew… |
| [OpenAI Whisper Tiny](https://huggingface.co/openai/whisper-tiny) | Hi. Hello. One, two, three. I'm John Wake. Pe… |
| [Facebook Wav2Vec2 Large](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) | HI HELLO ONE TWO THREE AMSON WAK PP |
| [Facebook HuBERT Large](https://huggingface.co/facebook/hubert-large-ls960-ft) | HIH HELLO ONE TWO THREE AMZON WAK PE PEW |
| [Distil-Whisper Medium](https://huggingface.co/distil-whisper/distil-medium.en) | Hi, hello. One, two, three. I'm Zon Wick. Pew… |

**Observations:**
- Whisper models output punctuation and normal capitalization, while Wav2Vec2 and HuBERT output uppercase text without punctuation.
- All five get "hi, hello, one two three" right and disagree on the name.
- This is one short clip with no reference transcript or word error rate, so it is a qualitative comparison only.

### **Text Generation**

**Notebook:** [text_generation.ipynb](./text_generation.ipynb)

**Model:** `pipeline("text-generation")` with no model named, which defaulted to [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

**Dataset:** [Robert Frost collection](https://www.kaggle.com/datasets/archanghosh/robert-frost-collection) from Kaggle, split into single lines of verse.

**Prompts tried:**
- The first and second lines of a poem, with `max_length` 20 and 30.
- The second line again with `num_return_sequences=2`.
- "What is an apple?" with `max_length=100`.

**Observations:**
- Continuations are plausible but unrelated to the real poem.
- The two returned sequences differ, so generation is not deterministic.
- The "What is an apple?" answer is cut off mid-sentence at `max_length`.

## **Setup**

The notebooks were run on Google Colab. The speech notebook uses `google.colab.files` to upload the audio file, so that part only works there as written.

Install the libraries used:

```bash
pip install transformers torch kagglehub pandas numpy scikit-learn seaborn matplotlib librosa soundfile
```

Notes:
- The sentiment and text generation notebooks download their datasets with `kagglehub`, which may ask you to authenticate with Kaggle.
- For the speech-to-text notebook, you need to supply your own audio clip and upload it, or replace the upload cell with a local file path.
- The first run downloads every model from the Hugging Face Hub, so it needs an internet connection.

---
[👆👆Back to top](#hugging-face-model-comparison)
