"""Final fifteen boundary-rule followups from fixed prediction-only endpoints.

The previous27 grid's best threshold.65 and minimumFGmean.8 were endpoint
settings. Test the available threshold.8 and lower.65 mean gates, always from
the original630 predictions. Never repeatedly refine a selected output.
"""
from pathlib import Path

import experiment_rich_boundary_refinement as engine


def plan():
    settings=[dict(name=f't{t:g}_fgmean{confidence:g}_guard{guard:g}',threshold=t,
                   minimum_foreground_mean=confidence,minimum_match_iou=.3,
                   minimum_original_iou=guard,area_ratio=[.5,1.5],fusion='2of3majority')
              for t,confidences in ((.8,(.85,.9,.95)),(.65,(.7,.75)))
              for confidence in confidences for guard in (.7,.85,.95)]
    assert len(settings)==15
    return settings


if __name__=='__main__':
    engine.plan=plan
    engine.EXPECTED_SETTINGS=15
    engine.REPORT_NAME='rich_boundary_followup.json'
    engine.OUT=Path('/workspace/artifacts/rich_boundary_refinement_followup_v1')
    engine.main()
