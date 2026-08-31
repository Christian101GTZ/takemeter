# TakeMeter

**TakeMeter** is an NLP text-classification project that analyzes discourse patterns in Reddit's r/Games community.

The project compares two approaches for classifying gaming posts into four categories:

* **Announcement**
* **Industry News**
* **Review / Critique**
* **Discussion**

I manually created and labeled a balanced dataset of **200 r/Games examples**, fine-tuned DistilBERT on the dataset, and compared its performance against a zero-shot Llama 3.3 70B baseline.

The experiment produced an unexpected result: the zero-shot LLM significantly outperformed the fine-tuned DistilBERT model. Rather than treating that as a failed project, I used the result to investigate dataset size, label overlap, classification errors, and the limitations of fine-tuning on small datasets.

---

## Project Goals

TakeMeter explores several practical NLP questions:

* Can a small transformer learn discourse categories from a manually labeled community-specific dataset?
* How does a fine-tuned DistilBERT model compare with a large zero-shot LLM?
* Which discourse categories are easiest or hardest to distinguish?
* How does annotation design affect downstream model performance?
* What can confusion matrices and incorrect predictions reveal about model behavior?

---

## Technologies

* Python
* Hugging Face Transformers
* DistilBERT
* Groq API
* Llama 3.3 70B
* Google Colab
* PyTorch
* Pandas
* scikit-learn
* CSV / JSON

---

## Dataset

The dataset was manually collected from public r/Games posts and discussion threads.

It contains **200 labeled examples**.

Each row contains:

```text
text,label,notes
```

* `text` — post title or summary
* `label` — assigned discourse category
* `notes` — explanation for the classification decision

The complete dataset is available in:

[`rgames_labeled_posts.csv`](./rgames_labeled_posts.csv)

### Label Distribution

| Label           | Examples |
| --------------- | -------: |
| Announcement    |       50 |
| Industry_News   |       50 |
| Review_Critique |       50 |
| Discussion      |       50 |
| **Total**       |  **200** |

The dataset was deliberately balanced so that no category would dominate the training data.

---

## Label Taxonomy

### Announcement

Posts whose primary purpose is announcing or revealing new content.

Examples include:

* Game announcements
* Trailers
* Release dates
* DLC
* Updates
* Seasons
* Showcases
* Launches

Example:

> Garfield: Escape from Monday Official Announcement Trailer reveals a new Garfield platformer release.

---

### Industry News

Posts primarily reporting factual information about the gaming industry.

Examples include:

* Company decisions
* Studio closures
* Layoffs
* Business results
* Legal disputes
* Hardware developments
* Technology
* Developer interviews
* Labor issues
* Market trends

Example:

> Nintendo suing the U.S. government over tariffs affecting its business operations.

---

### Review / Critique

Posts focused on evaluating games, mechanics, design choices, or player experiences.

Examples include:

* Review threads
* Retrospectives
* Long-form criticism
* Gameplay analysis
* Player impressions
* Performance analysis

Example:

> Bloodborne commentary analyzing boss design, progression, difficulty, storytelling, and gameplay systems.

---

### Discussion

Posts centered on community interaction rather than reporting or formal evaluation.

Examples include:

* Questions
* Recommendations
* Opinions
* Debates
* Genre discussions
* Gaming culture
* Personal experiences

Example:

> Can anyone recommend turn-based tactics games similar to XCOM that are not too difficult?

---

## Annotation Challenges

One of the most difficult parts of the project was defining categories that were meaningful while remaining distinct enough for classification.

### Discussion vs. Review / Critique

A community thread can contain detailed opinions about a game without necessarily being a formal review.

My rule was:

* General conversation, recommendations, or debate → **Discussion**
* Content primarily evaluating strengths and weaknesses → **Review_Critique**

### Industry News vs. Discussion

Industry events frequently generate community debate.

My rule was:

* Primarily reports an event, business decision, technical development, or company action → **Industry_News**
* Primarily framed around opinion or open-ended debate → **Discussion**

### Announcement vs. Review / Critique

Trailer posts often contain comments judging the game.

My rule was:

* The primary purpose of the post is a reveal, launch, update, or trailer → **Announcement**

The community reaction does not change the original purpose of the post.

---

## Training Approach

The fine-tuned classifier used:

```text
distilbert-base-uncased
```

Training was performed in Google Colab using a T4 GPU.

### Dataset Split

| Split      | Examples |
| ---------- | -------: |
| Training   |      140 |
| Validation |       30 |
| Test       |       30 |

### Training Configuration

| Parameter     | Value                     |
| ------------- | ------------------------- |
| Model         | `distilbert-base-uncased` |
| Epochs        | 3                         |
| Learning Rate | `2e-5`                    |
| Batch Size    | 16                        |

Because the dataset was relatively small, the training configuration remained close to the notebook defaults.

---

## Zero-Shot Baseline

Before evaluating the fine-tuned model, I created a zero-shot baseline using:

```text
llama-3.3-70b-versatile
```

through the Groq API.

The prompt provided:

* Definitions of all four labels
* Instructions to classify the post by its primary purpose
* A requirement to return only one valid category

The baseline and DistilBERT classifier were evaluated on the **same 30-example test set** to make the comparison consistent.

---

## Results

### Accuracy Comparison

| Model                   |  Accuracy |
| ----------------------- | --------: |
| Zero-Shot Llama 3.3 70B | **80.0%** |
| Fine-Tuned DistilBERT   | **53.3%** |

The zero-shot baseline outperformed the fine-tuned DistilBERT model by **26.7 percentage points**.

The experiment originally assumed that fine-tuning on community-specific examples would improve classification performance.

Instead, the results suggested that **200 examples were not enough for DistilBERT to reliably learn the boundaries between several overlapping discourse categories**.

The raw evaluation summary is also available in:

[`evaluation_results.json`](./evaluation_results.json)

---

## Per-Class Performance

### Fine-Tuned DistilBERT

| Label           | Precision | Recall |   F1 |
| --------------- | --------: | -----: | ---: |
| Industry_News   |      0.60 |   0.38 | 0.46 |
| Announcement    |      0.75 |   0.43 | 0.55 |
| Review_Critique |      0.44 |   0.88 | 0.58 |
| Discussion      |      0.60 |   0.43 | 0.50 |

### Zero-Shot Baseline

| Label           | Precision | Recall |   F1 |
| --------------- | --------: | -----: | ---: |
| Industry_News   |      1.00 |   0.75 | 0.86 |
| Announcement    |      0.78 |   1.00 | 0.88 |
| Review_Critique |      0.70 |   0.88 | 0.78 |
| Discussion      |      0.80 |   0.57 | 0.67 |

The baseline performed better across all four classes.

---

## Confusion Matrix

The fine-tuned DistilBERT confusion matrix was:

| True \ Predicted  | Industry News | Announcement | Review / Critique | Discussion |
| ----------------- | ------------: | -----------: | ----------------: | ---------: |
| Industry News     |             5 |            0 |                 3 |          0 |
| Announcement      |             0 |            7 |                 0 |          0 |
| Review / Critique |             3 |            0 |                 5 |          0 |
| Discussion        |             1 |            0 |                 5 |          1 |

![Fine-tuned DistilBERT confusion matrix](confustion_matrix.png)

The largest error pattern was the tendency to classify **Discussion** posts as **Review / Critique**.

Five Discussion examples were predicted as Review / Critique.

This suggests that the model learned to associate words related to:

* Gameplay quality
* Performance
* Balance
* Design
* Optimization
* Criticism

with the Review / Critique category, even when those words appeared inside open-ended community discussions.

---

## Error Analysis

### Industry News → Discussion

**Example**

> Nintendo suing the U.S. government over tariffs affecting its business operations.

**True label:** Industry_News
**Predicted:** Discussion
**Confidence:** 0.27

The example is clearly about an industry and legal event, but the text is extremely short.

The model may not have received enough contextual language to recognize the news-reporting purpose of the post.

---

### Announcement → Review / Critique

**Example**

> Garfield: Escape from Monday Official Announcement Trailer reveals a new Garfield platformer release.

**True label:** Announcement
**Predicted:** Review_Critique
**Confidence:** 0.28

The phrase **Official Announcement Trailer** strongly signals the correct category.

The model appears to have focused more heavily on surrounding game-description vocabulary instead of the announcement-related phrase.

---

### Discussion → Review / Critique

**Example**

> Players debated whether Planetside 2 should have been delayed, discussing optimization issues, game balance, launch expectations, and the challenges of releasing large-scale online games.

**True label:** Discussion
**Predicted:** Review_Critique
**Confidence:** 0.26

Terms such as *optimization*, *balance*, and *quality* resemble language used in reviews.

However, the primary purpose of the example is community debate rather than evaluating the game as a formal critique.

---

## What the Model Learned vs. What I Intended

My goal was for TakeMeter to classify posts according to their **primary communication purpose**.

The fine-tuned model instead appeared to rely heavily on surface-level vocabulary.

For example:

```text
"review" / "quality" / "balance" / "optimization"
             ↓
      Review_Critique
```

That works for obvious cases, but fails when similar words appear in community discussions or industry reporting.

The hardest distinctions were:

```text
Discussion ↔ Review_Critique

Industry_News ↔ Review_Critique
```

Announcement posts were generally easier because they frequently contain strong lexical indicators such as:

```text
announcement
trailer
release
launch
showcase
update
```

---

## What I Learned

The most important result of the project was not the final accuracy score.

The experiment demonstrated how heavily NLP performance can depend on:

* Dataset size
* Annotation consistency
* Label definitions
* Semantic overlap between classes
* Quality of training examples
* Evaluation methodology

Fine-tuning a model does not automatically guarantee better performance.

In this experiment, a much larger zero-shot LLM was able to interpret the broader semantic purpose of the posts more effectively than DistilBERT trained on only 140 examples.

The result reinforced the importance of comparing a trained model against a meaningful baseline instead of assuming fine-tuning is automatically an improvement.

---

## Limitations

### Small Dataset

The complete dataset contains only 200 examples, with 140 used for training.

This is a very small dataset for transformer fine-tuning and limits the model's ability to learn reliable decision boundaries.

### Small Test Set

The reported accuracy is based on only **30 test examples**.

The results demonstrate the behavior of this experiment but should not be interpreted as a comprehensive benchmark.

### Overlapping Categories

Discussion, Review / Critique, and Industry News sometimes contain extremely similar vocabulary.

The difference often depends on the **purpose of the post**, which is more difficult for a small fine-tuned model to infer from limited training examples.

### Manual Annotation

Labels were manually assigned using project-specific decision rules.

Although ambiguous examples were reviewed carefully, annotation remains subjective.

### Community-Specific Dataset

The dataset represents discourse from r/Games.

Performance should not be assumed to generalize to other Reddit communities or other kinds of online discussion.

---

## Future Improvements

Potential next steps include:

* Increase the dataset substantially beyond 200 examples
* Add more borderline examples between commonly confused classes
* Perform multiple train/test splits
* Use cross-validation
* Tune learning rate, epochs, and batch size
* Evaluate a larger transformer model
* Compare additional embedding or classical ML baselines
* Analyze model calibration
* Add confidence thresholds for uncertain predictions
* Experiment with merging or redesigning overlapping labels
* Evaluate performance on posts collected at a later date

A particularly important improvement would be increasing examples for:

```text
Discussion ↔ Review_Critique
```

because this was the most common source of model error.

---

## Repository Files

```text
takemeter/
├── rgames_labeled_posts.csv   # 200 manually labeled examples
├── evaluation_results.json    # Experiment summary
├── confustion_matrix.png      # Fine-tuned model confusion matrix
├── planning.md                # Taxonomy, annotation rules, and experiment planning
└── README.md                  # Project documentation
```

Training and evaluation were performed in Google Colab.

---

## Project Planning

The project was designed before final evaluation around specific performance goals.

The original targets included:

* At least 75% accuracy
* Macro F1 of at least 0.70
* No individual class below 0.60 F1
* Clear separation between all four categories

The fine-tuned model did not reach those targets.

Rather than changing the success criteria after seeing the results, I used the failed hypothesis as the basis for the error analysis and project reflection.

More detail is available in:

[`planning.md`](./planning.md)

---

## AI Usage

AI tools were used as assistants during several stages of the project.

### Label Design

ChatGPT was used to help stress-test the four-label taxonomy and identify examples that could reasonably fall into multiple categories.

This helped refine the decision rules between:

* Industry News and Discussion
* Discussion and Review / Critique
* Announcement and Review / Critique

Final label definitions were reviewed and selected manually.

### Dataset Review

AI assisted with reviewing potentially ambiguous examples and formatting some entries.

All examples in the final dataset were manually reviewed before their labels were accepted.

### Error Analysis

After training, AI was used to help identify recurring patterns among incorrect predictions.

The final conclusions were based on the actual evaluation results, confusion matrix, and manual review of the misclassified examples.

AI served as an analysis and development assistant rather than replacing the manual labeling or evaluation process.

---

## Demo

A video walkthrough of the project is available here:

[TakeMeter Demo](https://www.loom.com/share/22062a609c144d88af94e1554c878d14)

---

## Summary

TakeMeter demonstrates an end-to-end NLP experiment involving:

* Dataset design
* Manual annotation
* Label taxonomy development
* Transformer fine-tuning
* Zero-shot LLM classification
* Baseline comparison
* Precision, recall, and F1 evaluation
* Confusion-matrix analysis
* Error analysis
* Model limitation analysis

The fine-tuned DistilBERT classifier achieved **53.3% accuracy**, while the zero-shot Llama 3.3 70B baseline achieved **80.0%** on the same test set.

Although the fine-tuned model did not outperform the baseline, the result provided a useful case study in why dataset quality, dataset size, and label design can matter as much as model selection in applied NLP.
