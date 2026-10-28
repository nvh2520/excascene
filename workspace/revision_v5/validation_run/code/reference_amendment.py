"""Versioned reference corrections; no numeric or protocol changes."""
from common import *
import pandas as pd

locations = {
    'G03': 'main §8.3; Appendix Tables G.6–G.7 (S3-D/E)',
    'U03': 'main §7.1/Tables 8–9; Appendix Table F.2',
    'U05': 'Appendix Tables F.1–F.2; selector diagnostics',
    'U06': 'main §§3.2/4.5; oracle source',
    'U07': 'Appendix Table F.1; source edge-case audit',
    'U08': 'main §7.1; Appendix Table F.4',
    'U09': 'main §7.1/Table 9; Appendix Table F.2',
    'U10': 'Appendix Table F.5',
    'U11': 'Appendix Table F.6',
    'U12': 'Appendix Table F.6',
    'R01': 'main §6.2/Table 6; Appendix Tables E.2–E.4',
    'R02': 'main §6.2; Appendix Tables E.6–E.8',
    'D01': 'main §7; Appendix Table G.1',
    'D02': 'main §7.2; Appendix Table G.1',
    'D03': 'main §7.2/Table 10; Appendix Tables G.1/G.3',
    'C01': 'main §7.2; Appendix Table G.2',
    'C02': 'main §7.2; Appendix Table G.2',
    'C03': 'main §7.2; Appendix Table G.2',
    'C04': 'Appendix Table G.2; masking source',
    'N02': 'main §6.2/Figure 5; Appendix Table E.12',
    'N03': 'Appendix §A.2 and Tables F.2/G.2/G.7; statistics source',
    'N04': 'Appendix §I.6/Table I.7 and resampling source',
    'N05': 'Appendix §A.5/Tables A.3–A.4; evaluator source',
    'N06': 'main §5; renderer and metric source',
    'L01': 'main §3; Appendix §B.1/Table B.7',
    'L02': 'main §8.3; Appendix §B.1/Table B.7 and §I.6/Table I.6',
    'L04': 'main §3.3; Appendix Table B.5',
    'L05': 'main §§3.4/8.3/Table 2; Appendix §I.6/Tables I.6–I.8',
    'L06': 'main §§2/8.3; FACT-10c raw-recording evidence (Table E.14 is image-only context)',
    'R03': 'Appendix §§A.1–A.2/Tables A.1–A.2 and §J',
    'R05': 'main §5.1; Appendix §§I.1–I.5/J',
}

def main():
    locked()
    source = RUN/'claim_evidence_matrix_corrected.csv'
    old = pd.read_csv(source, keep_default_na=False)
    new = old.copy(); changes = []
    for claim, location in locations.items():
        i = new.index[new.claim_id.eq(claim)].item()
        before = new.loc[i, 'manuscript_location']
        new.loc[i, 'manuscript_location'] = location
        changes.append(dict(claim_id=claim, field='manuscript_location', old=before, new=location,
            reason='Final source/aux cross-check; distinguish table from subsection identifiers. G.6/G.7 contain S3-D/E; H is S3-C augmentation RNG. No numerical, model, population or statistical change.'))
    new.to_csv(RUN/'claim_evidence_matrix_corrected_v2.csv', index=False)
    csvwrite(RUN/'matrix_change_log_v2.csv', changes)
    paths = ['claim_evidence_matrix_corrected_v2.csv', 'matrix_change_log_v2.csv']
    js(RUN/'manifests/reference_amendment.json', dict(recorded_at=utc(),
        original_corrected_sha256=sha(source), files={p:sha(RUN/p) for p in paths},
        checked_sources={p:sha(ROOT/p) for p in ['paper_v4/main_v4.tex','paper_v4/appendix_v4.tex','paper_v4/main_v4.aux','paper_v4/appendix_v4.aux','paper_v4/sections/discussion.tex']},
        timing='After lock and during fresh inference; mechanical reference correction only. The locked original and first change log remain preserved.'))
    (RUN/'reports/erratum_references.md').write_text('# Versioned manuscript-reference correction\n\nThe first corrected matrix still assigned S3-D/E to Appendix H. Source and compiled auxiliary labels place them in Tables G.6–G.7; Appendix H concerns S3-C. Version 2 corrects this and distinguishes appendix table numbers from subsection numbers, removes unrelated training-configuration references from oracle/support claims, and identifies raw-recording evidence separately from image-only Table E.14. Every changed field is in matrix_change_log_v2.csv.\n\nThis correction occurred during fresh inference. It changes no numerical result, endpoint, model, exclusion or inference. The pre-outcome lock and its original matrix remain intact. Use claim_evidence_matrix_corrected_v2.csv and both change logs for the final crosswalk.\n')
    event('mechanical_reference_amendment', manifest='manifests/reference_amendment.json', claims=len(changes))
    print('Reference amendment:', len(changes), 'rows; original lock verified')

if __name__ == '__main__': main()
