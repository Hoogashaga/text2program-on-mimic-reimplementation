# Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records

This repository is the reproduction of the [Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records](https://arxiv.org/abs/2203.06918) (CHIL 2022).

## Abstract

This project reproduces the uncertainty-aware text-to-program model presented in the original paper for answering natural language questions on structured electronic health records (EHRs). The reproduction leverages a T5-based architecture to generate executable queries from natural language questions, capturing both data and model uncertainty to address ambiguity in healthcare queries, as described in the original research.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- python >= 8.0.0
- torch >= 2.0.0
- pytorch-lightning >= 2.0.0
- transformers >= 4.30.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- tqdm >= 4.65.0
- tensorboard >= 2.13.0
- scikit-learn >= 1.2.0
- rdflib >= 7.1.4
- datasets >= 2.12.0
- wandb >= 0.15.0

## Data Preparation

### Knowledge Graph Setup
1. Access MIMIC-III data (requires certification from [PhysioNet](https://mimic.physionet.org/))
2. Generate the MIMIC-III database:
   ```bash
   cd mimic_db_generator
   python process_mimic_db.py
   ```
   ref: https://github.com/wangpinggl/TREQS 

3. Build the knowledge graph for MIMICSPARQL* following these steps:
   ```bash
   cd MIMIC_SPARQL_KG_generator
   # First, build MIMICSQL* database from MIMICSQL database
   python build_mimicsparql_kg/build_mimicsqlstar_db_from_mimicsql_db.py
   
   # Then, build MIMIC-SPARQL knowledge graph from MIMICSQL* database
   python build_mimicsparql_kg/build_complex_kg_from_mimicsqlstar_db.py
   ```
   ref: https://github.com/junwoopark92/mimic-sparql)

4. The KG file (`mimic_sparqlstar_kg.xml`) should be placed in the `./data/db/mimicstar_kg` directory.

### Pre-processing
Generate dictionary files for the recovery technique:
```bash
cd data
python preprocess.py
```

## Training

To train the model with default configuration:

```bash
python main.py
```

For custom training configurations:
```bash
python main.py --model T5 \
               --train_batch_size 16 \
               --learning_rate 5e-5 \
               --resume_from_checkpoint "path to *.ckpt"\
```

## Evaluation

To evaluate the trained model:

```bash
python main.py --test
```

For ensemble model evaluation:
```bash
python main.py --model T5\
                --eval_batch_size 16\
               --ensemble_test \
               --ensemble_seed "42,1,12,123,1234" \
               --gpu_id "0"
```

## Pretrained Models

The reproduction of model is based on the Hugging Face T5 base pretrained model. Please refer to the [T5 model on Hugging Face](https://huggingface.co/t5-base) for details.


## Citation

```bibtex
@article{kim2022uncertainty,
  title={Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records},
  author={Kim, Daeyoung and Bae, Seongsu and Kim, Seungho and Choi, Edward},
  journal={arXiv preprint arXiv:2203.06918},
  year={2022}
}
```

The MIMIC-SPARQL knowledge graph generation builds upon:

```bibtex
@inproceedings{wang2020text,
  title={Text-to-SQL Generation for Question Answering on Electronic Medical Records},
  author={Wang, Ping and Shi, Tian and Reddy, Chandan K},
  booktitle={Proceedings of The Web Conference 2020},
  pages={350--361},
  year={2020}
}
```

## Related Work

This reproduction builds upon the following original works:

1. **Uncertainty-Aware Text-to-Program** (Original paper): [Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records](https://arxiv.org/abs/2203.06918) - CHIL 2022

2. **TREQS**: [Text-to-SQL Generation for Question Answering on Electronic Medical Records](https://github.com/wangpinggl/TREQS) - WWW'20

3. **MIMIC-SPARQL**: [Preparing MIMIC-III database and knowledge graph for SPARQL queries](https://github.com/junwoopark92/mimic-sparql)

## Contributing

We're specifically looking for contributions in the following areas:

1. **PyTorch Lightning compatibility**: 
   - Updates to support newer versions of PyTorch Lightning (2.0.0+)
   - Fixing deprecated API usage and adapting to new conventions
   
2. **Checkpoint management improvements**:
   - Enhanced checkpoint saving/loading functionality
   - Implementation of more robust resume-training capabilities
   
3. **Hyperparameter optimization**:
   - Tuning of patience parameters for early stopping
   - Learning rate schedule adjustments
   - Batch size optimization for modern GPUs

4. **Data processing integration**:
   - Consolidating data preprocessing scripts from multiple sources
   - Integrating MIMIC-III processing code into the main pipeline
   - Improving the efficiency of knowledge graph generation
   - Creating unified data processing utilities for both training and evaluation


## License

This project is licensed under the MIT License - see the LICENSE file for details.