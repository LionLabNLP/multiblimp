"""Conditions left out of a language's benchmark sample, set by hand (language folder name -> conditions, e.g.
{"Wolof": ["HEAD-DET_P"]}). The pairs stay in the pairs directory; build_benchmark skips them before sampling, so
the language refills from its other conditions, and counts them as config_excluded."""

benchmark_exclusions: dict = {}

# Candidates, not applied. Swapping Person on a possessive determiner or pronoun changes whose thing it is
# (seeni -> samay, "their" -> "my"); both sentences are grammatical, so the pair does not test agreement. The head
# noun's Person is not annotated and np_instance assumes 3, which is how these conditions pass the scan.
# Poss=Yes is no way to find them: it is missing for Wolof, and where it is set (Ruuli NounClass, Slavic and
# Norwegian Gender/Number/Case) the possessive agrees with its noun.
# Oct-30 benchmark, pairs per (condition, language):
#   HEAD-DET_P   Wolof 205, Khoekhoe 11, Bulgarian 6, German 6, Old_Church_Slavonic 5
#   HEAD-PRON_P  Hebrew 8 (possessive suffix), Turkish 8 (reflexive kendi)
# Not possessives, left in: Turkish HEAD-ADJ_P (107) and ADJ-ADP_P (3), an attributive adjective with a predicate
# person suffix (ungrammatical, but not agreement with the noun); HEAD-ADP_P (985), adpositions inflected for
# their object's person.
