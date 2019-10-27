# KGFP 
### Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization

This repository has code and datasets for KGFP, a system that predicts facts missing from a knowledge graph using embeddings computed via a knowledge-enriched tensor facorization approach.

The work is described in three papers:
* Ankur Padia, Konstantinos Kalpakis, Francis Ferraro and Tim Finin, [Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization](https://ebiquity.umbc.edu/paper/html/id/846/), Journal of Web Semantics, to appear, 2019.
* Ankur Padia, Konstantinos Kalpakis, Francis Ferraro and Tim Finin, [Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization](https://ebiquity.umbc.edu/paper/html/id/862/), International Semantic Web Conference, (journal track), October 2019.
* Ankur Padia, Kostantinos Kalpakis and Tim Finin, [Inferring Relations in Knowledge Graphs with Tensor Decompositions](https://ebiquity.umbc.edu/paper/html/id/766), IEEE Int. Conf. on Big Data, Dec. 2016.

KGFP developed and explored a family of four novel methods for embedding knowledge graphs into real-valued tensors that capture the ordered relations found in knowledge graphs including RDF graphs. Unlike many previous models, these can easily use prior background knowledge from users or extracted from existing knowledge graphs.

We demonstrated our models on the task of predicting new facts on eight different knowledge graphs, achieving a 5% to 50% improvement over existing systems. The data for these graphs in linked to or included in the repository.

We used eight datasets in our evaluation, including both previous graph-embedding benchmarks and new ones.
* **Kinship**: dataset with information about complex relational structure among 104 members of a tribe. It has 10,686 facts with 26 relations and 104 entities.
* **UMLS**: data on biomedical relationships between categorized concepts of the Unified Medical Language System. It has 6,752 facts with 49 relations and 135 entities.
* **WN18**: from WordNet [5], where entities are words that belong to synsets, which represent sets of synonymous words. Relations like hypernym, holonym, meronym and hyponym hold between the synsets. WN18 has 40,943 entities, 18 relations and more than 151,000 facts.
* **WN18RR**: dataset derived from WN18 that corrects some problems inherent in WN18 due to the large number of symmetric relations, which make it harder to create good training and testing datasets.  The dataset has 40,943 entities, 11 relations and more than 151,000 facts.
* **DB10k**: a real-world dataset with about 10,000 facts involving 4397 entities of type Person (e.g., Barack Obama) and 140 relations.
* **FB13**: a subset of a facts from Freebase with general information like ‘‘Johnny Depp won MTV Generation Award’’.  FB13 has 81,061 entities, 13 relationship and 360,517 facts.
* **FB15-237**: is a dataset derived from Freebase with about 310,000 facts about 237 relations and nearly 15K entities. It has triples coupled textual mention obtained from ClubWeb12.
* **FrameNet**: a lexical database describing how language can be used to evoke complex representations of Frames describing events, relations or objects and their participants.  The dataset has more than 62,000 facts about 16 relations among 22,000 entities.


Through experiments, we derived recommendations for selecting the best model based on knowledge graph characteristics. We also provide a provably-convergent, linear tensor factorization algorithm.

For more information, contact Dr. Ankur Padia, pankur1@umbc.edu.



