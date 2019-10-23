global ds_parameters
global sample_size
global maxIter
maxIter = 500


global conv
conv = 1e-6
ds_parameters = {
        "rescal": {
                     "dbpedia": { "lambda_A" : 10, "lambda_R" : 10},
                     "kinship": { "lambda_A" : 10, "lambda_R" : 10},
                     "umls": { "lambda_A" : 10, "lambda_R" : 10},
                     "wordnet": { "lambda_A" : 0.1, "lambda_R" : 0.1}, #{ "lambda_A" : 10, "lambda_R" : 10},
                     "framenet": {"lambda_A": 0.01, "lambda_R": 0.01},
                     "frame_trigger_lex_unit_pos_tag": {"lambda_A": 0.01, "lambda_R": 0.01},
                     "freebase": { "lambda_A" : 10, "lambda_R" : 10},
                     "fb13ntn": { "lambda_A" : 10, "lambda_R" : 10},
                     "wn11ntn": { "lambda_A" : 10, "lambda_R" : 10},
                     "framenet3": { "lambda_A" : 1, "lambda_R" : 1},
                     "framenet4": { "lambda_A" : 1, "lambda_R" : 1},
                     "wn18rr" : {"lambda_A" : 0.01, "lambda_R" : 0.002},
                     "fb15k_237": {"lambda_A": 0.01, "lambda_R": 0.002},
        },

        "quadratic_constraint": {
                     "dbpedia": { "lambda_A" : 100, "lambda_R" : 1000, "gemma":0, "alpha_a": 0.1, "alpha_r": 0.01, "alpha_lag_mult": 0.0001},
                     "kinship": {"lambda_A": 10, "lambda_R": 100, "gemma": 0, "alpha_a": 0.0001, "alpha_r": 0.0001, "alpha_lag_mult": 1.0},
                     "umls": {"lambda_A" : 10, "lambda_R" : 100,    "gemma":0, "alpha_a": 0.0001, "alpha_r": 0.0001, "alpha_lag_mult": 0.2},
                     "wordnet": { "lambda_A" : 100, "lambda_R" : 100, "gemma":0, "alpha_a": 0.1, "alpha_r": 0.01,"alpha_lag_mult": 0.0001},
                     "wn11ntn": { "lambda_A" : 100, "lambda_R" : 100, "gemma":0, "alpha_a": 0.1, "alpha_r": 0.01,"alpha_lag_mult": 0.0001},
                     "wn18rr": { "lambda_A" : 100, "lambda_R" : 100, "gemma":0, "alpha_a": 0.1, "alpha_r": 0.01,"alpha_lag_mult": 0.0001},
                     "fb15k_237": { "lambda_A" : 0, "lambda_R" : 0.001, "gemma":0, "alpha_a": 0.0001, "alpha_r": 0.001,"alpha_lag_mult": 0.001},
                     "freebase": { "lambda_A" : 100, "lambda_R" : 1000, "gemma":0, "alpha_a": 0.1, "alpha_r": 0.1, "alpha_lag_mult": 0.0001},
                     "fb13ntn": { "lambda_A" : 10, "lambda_R" : 1000, "gemma":0, "alpha_a": 0.1, "alpha_r": 0.1, "alpha_lag_mult": 0.001},
                     "framenet": { "lambda_A" : 1, "lambda_R" : 1, "gemma":0, "alpha_a": 1, "alpha_r": 0.1, "alpha_lag_mult": 0.0001},
                     "frame_trigger_lex_unit_pos_tag": { "lambda_A" : 1, "lambda_R" : 1, "gemma":0, "alpha_a": 1, "alpha_r": 0.1, "alpha_lag_mult": 0.0001}

        },

        "linear_constraint": {
                     "kinship": { "lambda_A" : 1, "lambda_E":10, "lambda_R" : 100, "gemma":0, "alpha":0.1, "alpha_a":0.01, "alpha_r":0.0001, "alpha_lag_mult":0.01},
                     "umls": {"lambda_A": 1,  "lambda_E": 10, "lambda_R": 100, "gemma": 0, "alpha": 0.1, "alpha_a": 0.00001, "alpha_r": 0.0001, "alpha_lag_mult": 0.1},
                     "dbpedia": { "lambda_A" : 10, "lambda_E":5, "lambda_R" : 1000, "gemma":0, "alpha_a": 0.1, "alpha_r": 0.01, "alpha_lag_mult": 0.0001},
                     "framenet": { "lambda_A" : 1,  "lambda_E":1, "lambda_R" : 1, "gemma":0, "alpha":0.1, "alpha_a":0.1, "alpha_r":0.001, "alpha_lag_mult":1},
                     "fb15k_237": { "lambda_A" : 1, "lambda_E":1, "lambda_R" : 1, "gemma":0, "alpha":0.1, "alpha_a":0.1, "alpha_r":0.001, "alpha_lag_mult":1},
                     "wordnet": {"lambda_A": 1000, "lambda_E": 1, "lambda_R": 1000, "gemma": 0, "alpha": 0.1, "alpha_a": 0.1, "alpha_r": 0.01, "alpha_lag_mult": 0.0001},
                     "wn18rr": {"lambda_A": 1000, "lambda_E": 1, "lambda_R": 1000, "gemma": 0, "alpha": 0.1, "alpha_a": 0.1, "alpha_r": 0.01, "alpha_lag_mult": 0.0001},
                     "freebase": { "lambda_A" : 10, "lambda_E": 5, "lambda_R" : 1000, "gemma":0, "alpha_a": 0.0001, "alpha_r": 0.1, "alpha_lag_mult": 0.0001},
        },

        "non_negative_rescal": {
             "dbpedia": { "lambda_A" : 0, "lambda_R" : 0},
             "kinship": { "lambda_A" : 0, "lambda_R" : 0},
             "umls": { "lambda_A" : 0, "lambda_R" : 0},
             "wordnet": { "lambda_A" : 0, "lambda_R" : 0},
             "freebase": { "lambda_A" : 0, "lambda_R" : 0},
             "fb13ntn": { "lambda_A" : 0.1, "lambda_R" : 0.1},
             "framenet": {"lambda_A": 10, "lambda_R": 10},
             "frame_trigger_lex_unit_pos_tag": {"lambda_A": 10, "lambda_R": 10},
             "framenet3": {"lambda_A": 10, "lambda_R": 10},
             'wn18rr' : {"lambda_A": 0.01, "lambda_R": 0.002},
             'wn11ntn' : {"lambda_A": 0.01, "lambda_R": 0.002},
        },

        "linear_regularized":{
             "dbpedia": { "lambda_A" : 0.01, "lambda_R" : 0.01, "lambda_sim" : 0.2, "lambda_E" : 10, "rho_inv" : 1},
             "umls": { "lambda_A" : 0.01, "lambda_R" : 0.01, "lambda_sim" : 0.00002, "lambda_E" : 10, "rho_inv" : 1},
             "kinship": {"lambda_A": 0.01, "lambda_R": 0.01, "lambda_sim": 0.00002, "lambda_E": 10, "rho_inv": 1},
             "wordnet": { "lambda_A" : 0.01, "lambda_R" : 0.01, "lambda_sim" : 0.00002, "lambda_E" : 10, "rho_inv" : 1},
             "wn18rr": { "lambda_A" : 0.01, "lambda_R" : 0.01, "lambda_sim" : 0.00002, "lambda_E" : 10, "rho_inv" : 1},
             "wn11ntn": { "lambda_A" : 0.01, "lambda_R" : 0.01, "lambda_sim" : 0.00002, "lambda_E" : 10, "rho_inv" : 1},
             "freebase": { "lambda_A" : 10, "lambda_R" : 0.2, "lambda_sim" : 0.00002, "lambda_E" : 1, "rho_inv" : 1},
             "fb13ntn": { "lambda_A" : 10, "lambda_R" : 0.2, "lambda_sim" : 0.00002, "lambda_E" : 1, "rho_inv" : 1},
             "framenet": { "lambda_A" : 0.01, "lambda_R" : 0.01, "lambda_sim" : 0.1, "lambda_E" : 1, "rho_inv" : 1},
        },

        "quadratic_regularized":{
             "dbpedia": { "lambda_A" : 0.0001, "lambda_R" : 0.002, "lambda_sim" : 0.2},
             "umls": { "lambda_A" : 0.1, "lambda_R" : 0.1, "lambda_sim" : 0.2},
             "kinship": {"lambda_A": 10, "lambda_R": 10, "lambda_sim": 0.2},
             "wordnet": { "lambda_A" : 0.1, "lambda_R" : 0.1, "lambda_sim" : 0.1},
             "wn18rr": { "lambda_A" : 0.1, "lambda_R" : 0.1, "lambda_sim" : 0.1},
             "wn11ntn": { "lambda_A" : 0.1, "lambda_R" : 0.1, "lambda_sim" : 0.1},
             "freebase": { "lambda_A" : 0.1, "lambda_R" : 0.1, "lambda_sim" : 0},
             "fb13ntn": { "lambda_A" : 0, "lambda_R" : 0, "lambda_sim" : 0},
             "framenet":{ "lambda_A" : 1, "lambda_R" : 10, "lambda_sim" : 1},
        }

}

sample_size = {

    "framenet": {
        "positive" : 6,
        "negative" : 4
    },
    "dbpedia" : {
        "positive" : 6,
        "negative" : 4
    },
    "kinship": {
        "positive": 6,
        "negative": 4
    },
    "umls" : {
        "positive" : 6,
        "negative" : 4
    },
    "wordnet" : {
        "positive" : 120,
        "negative" : 80
    },
    "freebase" : {
        "positive" : 120,
        "negative" : 80
    },
    "wn18rr" : {
        "positive" : 120,
        "negative" : 80
    },
    "wn11ntn" : {
        "positive" : 120,
        "negative" : 80
    },
    "fb13ntn" : {
        "positive" : 120,
        "negative" : 80
    },
    "fb15k_237" : {
        "positive" : 6,
        "negative" : 4
    }
}

