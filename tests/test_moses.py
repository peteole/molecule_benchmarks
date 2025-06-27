from molecule_benchmarks.moses_metrics import fingerprint, fingerprints, average_agg_tanimoto

def test_moses():
    print(fingerprint("CCO"))
    fps1 = fingerprints(["CCO", "CCN"])
    fps2 = fingerprints(["CCO", "CO"])
    print(fps1)
    print(fps2)
    print(average_agg_tanimoto(fps1, fps2))
    print(average_agg_tanimoto(fps1, fps1))