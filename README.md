# Privacy-Protect-Prompt-Obfuscation
We focus on prompt obfuscation by mixing the private prompts with virtual prompts, thereby preventing privacy leakage from both the input prompt and the LLM output.
## 1. Installation
```
git clone https://github.com/1006caomiao/Privacy-Protect-Prompt-Obfuscation.git
cd Privacy-Protect-Prompt-Obfuscation
```
## 2. Generate obfuscated samples
All the mentioned datasets have been processed and are placed in the `datasets` folder.

Generate obfuscated samples with customizable parameters:
```
python shuffling/generate_from_ner.py --parameter
```
## 3. Evaluation
We introduce two dedicated metrics: Binary Indistinguishability Rate (BIR) and Multi-Candidate Mixing Rate (MCMR).

We can conduct the evaluation through two methods: calling the API and deploying the model locally.

### Evaluate the BIR indicator:
```
python eval/apibir.py --parameter
```
or
```
python eval/localbir.py --parameter
```
### Evaluate the MCMR indicator:
```
python eval/apimcmr.py --parameter
```
or
```
python eval/localmcmr.py --parameter
```
