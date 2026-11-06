"""Reproduce own prediction gates, then associate external reference masks.

No neural inference, fitting, submission editing, or ground-truth scoring occurs.
The public reference is opened only after every own-image trace is finalized.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

import submission_yolo_recipe as yolo
import submission_native_recipe as native
from public_reference_audit import (
    SIZE, candidates, canonical, checked_mask, coverage, file_sha, ious, load_context,
    load_image, names_from, row_masks, write_csv,
)
from public_reference_stage_trace import replay_stages
from reference_panel_audit import load_panel, REFERENCE_SHA
from filament_eval import read_csv


def portable(mask):
    return dict(size=list(mask['size']), counts=mask['counts'].decode() if isinstance(mask['counts'], bytes) else mask['counts'])


def mask_sha(mask):
    counts=mask['counts'].encode() if isinstance(mask['counts'],str) else mask['counts']
    return hashlib.sha256(counts).hexdigest()


def parity(left, right, name):
    if [mask_sha(r) for r in left] != [mask_sha(r) for r in right]:
        raise ValueError(name+' trace differs from frozen recipe')


def trace_preprocessing(originals, geometric, preprocessing):
    """Instrument the exact frozen ordering, admission gates and occupancy."""
    output=list(originals)
    covered=mu.decode(mu.merge(output)).astype(bool) if output else np.zeros((SIZE,SIZE),bool)
    old=[yolo.native_rle(p) for record in geometric.values() for p in record['instances'] if p['confidence']>=.1]
    traces=[]; pending=[]
    for view,record in preprocessing.items():
        for index,proposal in enumerate(record['instances']):
            mask=yolo.native_rle(proposal); area=int(mu.area(mask))
            item=dict(source='preprocessing',view=view,index=index,confidence=float(proposal['confidence']),
                area=area,mask_sha256=mask_sha(mask),mask=portable(mask),edge_clipped=False,
                geometry_max_iou=None,occupied_fraction=None,remaining_area=None,processing_order=None)
            if view=='limb':item['route_reason']='limb_view_excluded_from_rescue'
            elif proposal['confidence']<.2:item['route_reason']='below_rescue_pool_confidence_0_2'
            else:pending.append((proposal,mask,item))
            traces.append(item)
    pending.sort(key=lambda x:-x[0]['confidence'])
    matrix=ious([r[1] for r in pending],old)
    for order,(proposal,mask,item) in enumerate(pending):
        item['processing_order']=order
        agreement=float(matrix[order].max()) if old else 0.
        item['geometry_max_iou']=agreement
        # These simultaneous static gates use the same short-circuit order as
        # the frozen recipe. Later occupancy is meaningful only after all pass.
        failures=[]
        if proposal['confidence']<.3:failures.append('confidence_below_0_3')
        if item['area']<300:failures.append('area_below_300')
        if agreement<.3:failures.append('geometric_agreement_below_0_3')
        item['static_failed_gates']='|'.join(failures)
        if failures:item['route_reason']=failures[0];continue
        x,y,w,h=map(int,mu.toBbox(mask))
        local=mu.decode(mask)[y:y+h,x:x+w].astype(bool)
        occupied=covered[y:y+h,x:x+w]
        fraction=int((local&occupied).sum())/item['area']
        item['occupied_fraction']=fraction
        if fraction>=.1:item['route_reason']='occupied_fraction_at_least_0_1';continue
        remaining=local&~occupied
        item['remaining_area']=int(remaining.sum())
        if item['remaining_area']<300:item['route_reason']='remaining_area_below_300';continue
        occupied|=remaining
        accepted=yolo.encode_roi(remaining,y,x)
        output.append(accepted)
        item['route_reason']='admitted_to_preprocessing_output'
        item['admitted_mask_sha256']=mask_sha(accepted)
    parity(output,yolo._preprocessing_rescue(originals,geometric,preprocessing),'preprocessing rescue')
    return output,traces


def trace_own_image(stem, stages, baseline, expected_source_sha256):
    """Prediction-only trace. No reference, labels, names, or file IO arguments."""
    selected=replay_stages(stem,stages,baseline,expected_source_sha256=expected_source_sha256)
    geometric=yolo._view_map(stages['geometric'],yolo.GEOMETRIC_VIEWS)
    preprocessing=yolo._view_map(stages['preprocessing'],yolo.PREPROCESSING_VIEWS)
    votes=np.zeros((SIZE,SIZE),np.uint8)
    traces=[]
    for view,record in geometric.items():
        eligible=[]
        for index,p in enumerate(record['instances']):
            mask=yolo.native_rle(p)
            reason=('view_not_used_in_primary_consensus' if view=='hflip' else
                    'confidence_below_vote_0_2' if p['confidence']<.2 else
                    'contributes_to_pixel_vote_not_direct_instance_admission')
            traces.append(dict(source='geometric',view=view,index=index,confidence=float(p['confidence']),
                area=int(mu.area(mask)),mask_sha256=mask_sha(mask),mask=portable(mask),edge_clipped=False,route_reason=reason))
            if view in ('base','scale','vflip') and p['confidence']>=.2:eligible.append(mask)
        if eligible:votes+=mu.decode(mu.merge(eligible))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed=cv2.morphologyEx((votes>=2).astype(np.uint8),cv2.MORPH_CLOSE,kernel,
        borderType=cv2.BORDER_CONSTANT,borderValue=0)
    before_area=yolo._component_rles(closed,1)
    after_area=[r for r in before_area if int(mu.area(r))>=300]
    parity(after_area,selected['yolo_consensus'],'geometric consensus')
    pre_out,pre_trace=trace_preprocessing(after_area,geometric,preprocessing)
    parity(pre_out,selected['yolo_preprocessing_rescue'],'preprocessing stage')
    traces+=pre_trace
    for record in stages['tiled']:
        for index,p in enumerate(record['instances']):
            mask=yolo.native_rle(p)
            traces.append(dict(source='tiled',view=record['view'],index=index,confidence=float(p['confidence']),
                area=int(mu.area(mask)),mask_sha256=mask_sha(mask),mask=portable(mask),
                edge_clipped=bool(p.get('edge_clipped',False)),
                route_reason='tiled_shapes_refine_existing_instances_no_direct_new_object_admission',
                tile_refinement_basic_eligible=bool(not p.get('edge_clipped',False) and p['confidence']>=.35),
                tile_refinement_scope='Basic eligibility only; assignment, paired scale and shape guards also apply'))
    rich=stages['native']; originals=selected['native_rich_refined']
    views=[native._rle(v) for v in rich['view_unions'].values()]
    count=sum((mu.decode(v) for v in views),np.zeros((SIZE,SIZE),np.uint8))
    stable=mu.encode(np.asfortranarray((count>=2).astype(np.uint8)))
    support=native._rle(rich['tta']['unions']['foreground']['0.2'])
    prune=[]
    for index,mask in enumerate(originals):
        foreground=coverage(mask,support); view_support=coverage(mask,stable)
        dropped=foreground<.4 and view_support<.25
        prune.append(dict(index=index,mask_sha256=mask_sha(mask),mask=portable(mask),
            area=int(mu.area(mask)),native_tta020_coverage=foreground,native_stable2_coverage=view_support,
            pruned=dropped,route_reason='both_native_support_gates_failed' if dropped else 'kept_by_native_support'))
    final,counters=native._final_admission_pruning(originals,rich)
    parity(final,baseline,'final admission/pruning')
    if counters['final_removed']!=sum(r['pruned'] for r in prune):raise ValueError('Pruning trace count mismatch')
    return dict(image=stem,own_stage_parity=True,raw_yolo=traces,
        raw_view_metadata={source:[{k:v for k,v in record.items() if k!='instances'}
                                  for record in stages[source]]
                           for source in ('geometric','preprocessing','tiled')},
        geometry_components_before_area=[portable(r) for r in before_area],
        stages={name:[portable(r) for r in masks] for name,masks in selected.items()},
        final_pruning=prune,final_counters=counters,
        trace_scope='Exact preprocess admission and final original-mask pruning; geometry vote eligibility and area gate. Tiled assignment/native-component admission causes are not fully instrumented.')


def join_reference(stem, trace, references):
    """Diagnostic association after prediction-only trace has been frozen."""
    raw=trace['raw_yolo']; raw_masks=[checked_mask(r['mask']) for r in raw]
    ref_masks=[m for _,m in references]
    matrix=ious(ref_masks,raw_masks)
    stages={name:ious(ref_masks,[checked_mask(m) for m in masks]) for name,masks in trace['stages'].items()}
    pre_area=ious(ref_masks,[checked_mask(m) for m in trace['geometry_components_before_area']])
    prune=trace['final_pruning']; prune_iou=ious(ref_masks,[checked_mask(r['mask']) for r in prune])
    targets=[]; candidate_rows=[]; prune_rows=[]
    for i,(identifier,_) in enumerate(references):
        matching=[j for j in range(len(raw)) if matrix[i,j]>.5]
        selected=[raw[j] for j in matching]
        row=dict(image=stem,reference_id=identifier,matching_yolo_candidates=len(selected),
            max_confidence_among_matching_yolo=max((r['confidence'] for r in selected),default=None),
            matching_yolo_source_counts=json.dumps(dict(Counter(r['source'] for r in selected)),sort_keys=True),
            matching_yolo_route_reason_counts=json.dumps(dict(Counter(r['route_reason'] for r in selected)),sort_keys=True))
        for threshold in (.1,.15,.2,.3):
            row['matching_yolo_conf_ge_'+str(threshold).replace('.','_')]=sum(r['confidence']>=threshold for r in selected)
        for name,values in stages.items():row[name+'_max_iou']=float(values[i].max()) if values.shape[1] else 0.
        row['consensus_before_area_max_iou']=float(pre_area[i].max()) if pre_area.shape[1] else 0.
        row['consensus_similarity_lost_only_by_area300']=(row['consensus_before_area_max_iou']>.5 and row['yolo_consensus_max_iou']<=.5)
        for j in matching:
            item={k:v for k,v in raw[j].items() if k!='mask'}
            candidate_rows.append(dict(image=stem,reference_id=identifier,reference_iou=float(matrix[i,j]),**item))
        old_matches=[j for j in range(len(prune)) if prune_iou[i,j]>.5]
        row['similar_old_masks_pruned']=sum(prune[j]['pruned'] for j in old_matches)
        for j in old_matches:
            prune_rows.append(dict(image=stem,reference_id=identifier,reference_iou=float(prune_iou[i,j]),
                **{k:v for k,v in prune[j].items() if k!='mask'}))
        targets.append(row)
    return targets,candidate_rows,prune_rows


def run(cache_root, reference_csv, expected_names, panel216, panel102, output_root):
    root=Path(cache_root).resolve(); output=Path(output_root).resolve()
    if output.exists() or output==root or root in output.parents or output in root.parents:
        raise ValueError('Use a new trace output folder separate from inference inputs')
    snapshot={Path(__file__).resolve():file_sha(__file__)}
    protocol,identity,runtime,names=load_context(root,snapshot)
    if names!=names_from(expected_names) or len(names)!=180:raise ValueError('Expected complete official test180')
    paths=[Path(expected_names),Path(panel216),Path(panel102)]
    for path in paths:snapshot[path]=file_sha(path)
    panels={216:load_panel(panel216,216),102:load_panel(panel102,102)}
    output.mkdir(parents=True); own=output/'own_trace';own.mkdir()
    images={r['stem']:r for r in protocol['images']}
    for index,stem in enumerate(names,1):
        stages,baseline,final=load_image(root,images[stem],identity,runtime,snapshot)
        candidates(stem,stages)  # Validate every raw RLE before COCO operations.
        trace=trace_own_image(stem,stages,baseline,protocol['sources'])
        trace['cache_identity']=identity
        trace['r1_masks']=[portable(r) for r in final]
        (own/(stem+'.json.gz')).write_bytes(gzip.compress(canonical(trace),compresslevel=6,mtime=0))
        if index%20==0 or index==len(names):print(f'Own gate trace {index}/{len(names)}',flush=True)
    # Public mask content is first opened here, after all own-stage decisions.
    snapshot[Path(reference_csv)]=file_sha(reference_csv)
    if snapshot[Path(reference_csv)]!=REFERENCE_SHA:raise ValueError('Public reference SHA changed')
    reference=row_masks(read_csv(reference_csv)); wanted={r['reference_id'] for rows in panels.values() for r in rows}
    target_rows=[];candidate_rows=[];prune_rows=[]
    for stem in names:
        refs=[r for r in reference.get(stem,[]) if r[0] in wanted]
        if not refs:continue
        trace=json.loads(gzip.decompress((own/(stem+'.json.gz')).read_bytes()))
        target,matching_candidates,prune=join_reference(stem,trace,refs)
        target_rows+=target;candidate_rows+=matching_candidates;prune_rows+=prune
    if len(target_rows)!=318 or {r['reference_id'] for r in target_rows}!=wanted:raise ValueError('Missing fixed panel target')
    write_csv(output/'matching_candidate_gates.csv',candidate_rows)
    write_csv(output/'matching_final_pruning.csv',prune_rows)
    for count,panel in panels.items():
        ids={r['reference_id'] for r in panel}
        write_csv(output/f'reference{count}_gate_summary.csv',[r for r in target_rows if r['reference_id'] in ids])
    for path,expected in snapshot.items():
        if file_sha(path)!=expected:raise ValueError('Input changed during trace: '+str(path))
    own_hashes={p.name:file_sha(p) for p in sorted(own.glob('*.json.gz'))}
    summary=dict(status='complete',images=len(names),cache_identity=identity,
        scope='External reference association only; no test GT, PQ, TP, FP or FN.',
        candidate_rle_export=True,raw_export_scope='All own YOLO30 views, own intermediate masks and pruning trace; native raw component inventory is not exported.',
        geometry_scope='Raw vote eligibility is not a direct instance acceptance/rejection decision; union voting, closing and connected-component formation transform shape.',
        preprocessing_scope='Exact execution gate/order/occupancy, with frozen output parity.',
        native_scope='Exact final keep/drop of existing rich-refined masks. Additional native-component gates are not fully traced.',
        reference_used_to_select_own_predictions=False,own_trace_frozen_before_reference_content_loaded=True,
        matching_candidate_rows=len(candidate_rows),matching_pruning_rows=len(prune_rows),
        input_sha256={str(p):v for p,v in snapshot.items()},own_trace_sha256=own_hashes,
        source_sha256=file_sha(__file__),neural_inference=False,training=False,submissions_modified=False)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('Trace complete: references318; raw YOLO candidate rows',len(candidate_rows),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','reference-csv','expected-names','panel216','panel102','output-root'):
        parser.add_argument('--'+name,required=True,type=Path)
    run(**vars(parser.parse_args()))


if __name__=='__main__':main()
