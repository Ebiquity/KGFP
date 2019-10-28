# KGFP 
### Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization

This repository has code and data the knowledge graph embeddings techniques presented in the paper "Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization" (Padia et al., 2019). Please see the bibtex link below for the full citation.  

KGFP developed and explored a family of four novel methods for embedding knowledge graphs into real-valued tensors that capture the ordered relations found in knowledge graphs including RDF graphs. Unlike many previous models, these do not need access to a semantic schema and can easily use prior background knowledge from users or extracted from existing knowledge graphs.  Through experiments, we derived recommendations for selecting the best model based on knowledge graph characteristics. We also provide a provably-convergent, linear tensor factorization algorithm.

For more information, contact Dr. Ankur Padia, pankur1@umbc.edu.

# Datasets #
There are eight frequently benchmarked datasets. Each dataset contains set of positive and negative examples.

| Name     | # Entities | # Relations | # Facts | Avg. Deg. | Graph Density |
| -------- | ---------- | ----------- | ------- | --------- | ------------- |
| Kinship  |   104  | 26     | 10,686 | 102.75 | 0.98798  |
| UMLS     |   135  | 49  | 6,752 | 50.01 | 0.37048 |
| FB15-237 | 14,541 | 237 | 310,116 | 21.32 | 0.00147 |
| DB10k    |  4,397 | 140    | 10,000 | 2.27 | 0.00052 |
| FrameNet | 22,298 | 16 | 62,344 | 2.79 | 0.00013 |
| WN18     | 40,943 | 18 | 151,442 | 3.70 | 0.00009 |
| FB13     | 81,061 | 13 | 360,517 | 4.45 | 0.00005 |
| WN18RR   | 40,943 | 11    | 93,003 | 2.27 | 0.00005 |


- **Kinship** is dataset with information about complex relational structure among 104 members of a tribe. It has 10,686 facts with 26 relations and 104 entities. From this, we created a tensor of size 104x104x26.

- **UMLS** has data on biomedical relationships between categorized concepts of the Unified Medical Language System. It has 6,752 facts with 49 relations and 135 entities. We created a tensor of size 135x135x49.

- **WN18** contains information from WordNet, where entities are words that belong to synsets, which represent sets of synonymous words. Relations like hypernym, holonym, meronym and hyponym hold between the synsets.

- **WN18** has 40,943 entities, 18 different relationships and more than 151,000 facts. We created a tensor of size 40,943x40,943x18.

- **WN18RR** is a dataset derived from WN18 that corrects some problems inherent in WN18 due to the large number of symmetric relations. These symmetric relations make it harder to create good training and testing datasets. For example, a training set might contain (e1; r1; e2) and test might contain its inverse (e2; r1; e1), or a fact occurring with e1ande2 with some relation r2.

- **FB13** is a subset of a facts from Freebase that contains general information like "Johnny Depp won MTV Generation Award". FB13 has 81,061 entities, 13 relationship and 360,517 facts. We created a tensor of size 81,061x81,061x13.

- **FrameNet** is a lexical database describing how language can be used to evoke complex representations of Frames describing events, relations or objects and their participants.  For example, the Commerce buy frame represents the interrelated concepts surrounding stereotypical commercial transactions. Frames have roles for expected participants (e.g., Buyer, Goods, Seller), modifiers (e.g., Imposed purpose and textttPeriod of iterations), and inter-frame relations defining inheritance and usage hierarchies (e.g., Commerce buy inherits from the more general Getting and is inherited by the more specific Renting.  We processed FrameNet 1.7 to produce triples representing these frame-to-frame, frame-to-role, and frame-toword relationships. FrameNet 1.7 defines roughly 1,000 frames, 10,000 lexical triggers, and 11,000 (frame-specific) roles. In total, we used 16 relations to describe the relationship among these items.

- **DB10k** is a real-world dataset with about 10,000 facts involving 4,397 entities of type Person (e.g., Barack Obama) and 140 relations. We used a DBpedia public SPARQL endpoint to collect the facts which were processed in the following manner. When the object value was a date or number, we replaced the object value with fixed tag.  For example, "Barack Obama marriedOn 1992-10-03 (xsd:date)" is processed to produce "Barack Obama marriedOn date". In case object is an entity it is left unchanged. For example "Barack Obama is-a President" as President is an entity. Such an assumption can strengthen the overall learning process as entities with similar attribute relations will tend to have similar value in the tensor. After processing, a tensor of size 4397x4397x140 was created.

- **FB15-237** is a dataset containing subset of the Freebase with 237 relations and nearly 15K entities. It has triples coupled textual mention obtained from ClubWeb12.


# Models #
*KGFP/models* contains implementation of all the models discussed in the journal manuscript. Models which are linear in the  number of entities are implemented in **linear\_regularized.py** and **linear\_constraint.py**, and the models which are  quadratic in  the number of entities are implemented in **quadratic\_regularized.py** and  **quadratic\_constraint.py**.

# Requirements #

- scipy (1.1.0)
- scikit-learn (0.19.1)
- tabulate (0.8.3)

To install each of the above requirements run the following commands:

<pre>
<code>
python -m pip install --user numpy scipy
pip install -U scikit-learn
pip install -U tabulate
</code>
</pre>


# Train and Test a model #
To train and test a model run the following command:

<pre>
<code>
python experiments/experiments.py --algo quadratic_regularized --dataset kinship --distance transitivity
</code>
</pre>

If you find issues related to missing modules add PYTHONPATH as follows

<pre>
<code>
PYTHONPATH='/media/a/bigdata/git/KGFP' python experiments/experiments.py --algo quadratic_constraint--dataset kinship --distance transitivity
</code>
</pre>

Value of the *PYTHONPATH* is set the location of KGFP folder and makes the required package/modules available to python interpreter. 

Here 

- **--dataset** can take a value from {kinship, umls, wordnet, dbpedia, freebase, framenet, wn18rr, fb15k_237}. 
- **--distance** can take a value from {*agency, transitivity, reverse_transitivity, patient*}
- **--algo** can take a value from {*quadratic_constraint, quadratic_regularized, linear_constraint, linear_regularized*}

# Output and Running time #

Output will contains the performance for all datasets across all metric. The larger and longer the tensor more time it will take to run the model(s). 

# Papers #
The work is described in three papers.  The first is the most complete version, the second is a shorter journal-track paper version from the ISWC conference, and the third is an early, preliminary paper that explored some of the ideas.
* Ankur Padia, Konstantinos Kalpakis, Francis Ferraro and Tim Finin, [Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization](https://ebiquity.umbc.edu/paper/html/id/846/), Journal of Web Semantics, to appear, 2019.
* Ankur Padia, Konstantinos Kalpakis, Francis Ferraro and Tim Finin, [Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization](https://ebiquity.umbc.edu/paper/html/id/862/), International Semantic Web Conference, (journal track), October 2019.
* Ankur Padia, Kostantinos Kalpakis and Tim Finin, [Inferring Relations in Knowledge Graphs with Tensor Decompositions](https://ebiquity.umbc.edu/paper/html/id/766), IEEE Int. Conf. on Big Data, Dec. 2016.

## Cite ##

If you use our model(s) or want to refer to it in a paper, please cite:

<pre>
Padia, Ankur, Konstantinos Kalpakis, Francis Ferraro and Tim Finin, Knowledge graph fact prediction via knowledge-enriched tensor factorization, Journal of Web Semantics, Elsevier, 2019.
</pre>

## BibTeX ##

<pre>
<code>
@Article{Knowledge_Graph_Fact_Prediction_via_Knowledge_Enriched_Tensor_Factorization,
       author = "Ankur Padia and Kostantinos Kalpakis and Francis Ferraro and Tim Finin",
       title = "{Knowledge Graph Fact Prediction via Knowledge-Enriched Tensor Factorization}",
       month = "February",
       year = "2019",
       journal = "Journal of Web Semantics",
       publisher = "Elsevier",
}         
</code>
</pre>



