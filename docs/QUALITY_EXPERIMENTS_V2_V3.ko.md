# 마스크 품질 기반 독창적 후처리 실험 v2/v3

707장·1,154주석그룹·8,199객체에서43개 설정을 실제 비교했다. 같은 결과인 설정과
개발 결과를 본 뒤 선택한 부품 조합도 포함된다. 추가 학습·신경망 추론은 없었다.
Native는5개fold모두30epoch, YOLO는30/30/30/30/25epoch로 고정했다.

## 현재 최고 개발 후보

| 항목 | 고정 기준 | 새 후보 | 변화 |
|---|---:|---:|---:|
| PQ | 0.431873883 | **0.436752508** | **+0.004878626** |
| TP | 5187 | 5244 | +57 |
| FP | 2919 | **2891** | **−28** |
| FN | 3012 | **2955** | **−57** |

GT101개를 복구하고44개 매칭을 잃어 순FN57개가 감소했다. 새 후보는직전FN우선C4
(PQ0.435294610)보다도FP·FN이각각11개감소했다. 저장된CSV를별도프로세스에서다시읽어
동일한PQ·SQ·RQ·TP·FP·FN을독립적으로재현했다.

- 5개fold·6개관측소 모두PQ상승. 개별fold마다FP와FN이둘다줄어든것은아니다.
- 원본대비조건부physical-image bootstrap95% Δ구간[+0.003075,+0.006723].
- 직전C4대비조건부구간[+0.000438,+0.002444].
- 같은OOF를여러번비교하고부품을선택한결과이므로선택편향·시간상관이보정된
  독립적인성능확정이아니다. 새publicLB점수는없으며목표PQ0.7에는아직미달이다.

## 효과를 확인한 구성

1. **마스크 품질 선택:** class confidence보다native base/TTA의여러threshold마스크와
   모양이잘맞는raw YOLO후보로기존객체의모양을교체한다. 원본IoU·면적·연결성분·침입을제한한다.
2. **객체 병합:** 여러geometric view와native의동일전경component가같은객체로지지할때
   잘린두조각을병합한다. 이단계자체는foreground union을그대로유지한다.
3. **보수적 제거:** 면적≤1000이며native신뢰영역과YOLO원시후보의지지가함께약한객체를제거한다.
4. **강한 중심부 기반 복구:** 여러YOLO계열이지지하고native고확률core·전경범위가
   확인된새후보를추가한다. 제거전모든원본·교체영역을예약해후보를클리핑한다.
5. **4픽셀 경계 hysteresis:** 두native가모두0.2미만인경계만줄이고둘다0.65이상인
   외곽만늘린다. 불확실한영역은원본유지,면적비0.8–1.25·원본IoU≥0.85·연결성분·충돌guard를적용한다.

R/M/P/A의결정은immutable원래예측에서계산하고,교체R→원본단위keep P→M그룹union→
예약영역밖A추가순서로조합한다. 마지막경계수정은그결과에적용한다. 데이터전체에서
똑같은규칙을사용하며정답객체별로규칙을고르지않았다.

## 유지하지 않은 방법

- YOLO view끼리의모양일치도만으로마스크를선택하면개선되지않았다.
- Family별대표1개로균형을맞춘다수결·가중투표도모두기준미달이었다. 서로틀린경계를
  평균내는것보다교차모델품질근거로좋은경계를선택하는쪽이효과적이었다.
- Raw후보로기존객체를더많이분할·재조립하면복구와함께FP·기존매칭손실이늘었다.
- Native고확률신호를사용한단독추가는정밀도를개선했지만그자체로PQ상승에는미달했다.
- 양자화native확률의평균·주변대비및1–2픽셀이동만으로는큰개선이없었다.
  이지표들은복원된연속확률이나실제GT-IoU확률이아닌heuristic이다.

## 전체 설정 기록

| 계열 / 설정 | PQ | FP | FN |
|---|---:|---:|---:|
| current_native_core_rescue_v2 / N1_tta05_half | 0.431708124 | 3071 | 2942 |
| current_native_core_rescue_v2 / N2_tta05_seventy | 0.431601685 | 3039 | 2960 |
| current_native_core_rescue_v2 / N3_tta065_half | 0.431579622 | 3056 | 2952 |
| current_native_core_rescue_v2 / N4_core_and_extent | 0.431828289 | 3018 | 2966 |
| current_native_core_rescue_v2 / N5_lowconf_two_branch | 0.430583777 | 3222 | 2891 |
| current_native_core_rescue_v2 / N6_lowconf_core | 0.430492676 | 3225 | 2891 |
| current_native_core_rescue_v2 / N7_native_quality_mean | 0.430977494 | 3172 | 2908 |
| current_native_core_rescue_v2 / N8_native_ring_contrast | 0.429983440 | 3263 | 2882 |
| current_mask_quality_replacement_v2 / agreement_strict | 0.431407299 | 2919 | 3012 |
| current_mask_quality_replacement_v2 / agreement_large_gain | 0.431710129 | 2922 | 3015 |
| current_mask_quality_replacement_v2 / native_strict | 0.432941027 | 2909 | 3002 |
| current_mask_quality_replacement_v2 / native_large_gain | 0.433223249 | 2907 | 3000 |
| current_mask_quality_replacement_v2 / hybrid_strict | 0.432100707 | 2917 | 3010 |
| current_mask_quality_replacement_v2 / hybrid_large_gain | 0.431915285 | 2921 | 3014 |
| current_corroborated_topology_v2 / S1_geo2_native065_split | 0.431323886 | 2964 | 2999 |
| current_corroborated_topology_v2 / S2_family2_native08_split | 0.429321688 | 3030 | 2998 |
| current_corroborated_topology_v2 / M1_geo2_native035_merge | 0.432755303 | 2891 | 3010 |
| current_corroborated_topology_v2 / M2_family2_native035_merge | 0.432688097 | 2886 | 3014 |
| current_probability_quality_v2 / Q1_weak_low_contrast | 0.431873883 | 2919 | 3012 |
| current_probability_quality_v2 / Q2_larger_weak_low_contrast | 0.431873883 | 2919 | 3012 |
| current_probability_quality_v2 / Q3_offcenter_native_signal | 0.431873883 | 2919 | 3012 |
| current_probability_quality_v2 / Q4_low_cross_model_quality | 0.431895491 | 2908 | 3017 |
| current_probability_quality_v2 / Q5_both_native_and_yolo_shape_weak | 0.432032864 | 2913 | 3012 |
| current_probability_quality_v2 / Q6_P1_protect_contrast_core | 0.431883306 | 2917 | 3013 |
| current_probability_quality_v2 / T1_native_agreed_one_pixel_shift | 0.431949019 | 2918 | 3011 |
| current_probability_quality_v2 / T2_native_agreed_guarded_shift | 0.431883120 | 2919 | 3012 |
| current_quality_composition_v2 / C1_R_M | 0.434106824 | 2879 | 2998 |
| current_quality_composition_v2 / C2_R_M_P | 0.435371577 | 2803 | 3012 |
| current_quality_composition_v2 / C3_R_M_A | 0.434041445 | 2978 | 2952 |
| current_quality_composition_v2 / C4_R_M_P_A | 0.435294610 | 2902 | 2966 |
| current_object_cluster_fusion_v3 / F1_family_medoid_majority_strict | 0.427702690 | 2940 | 3033 |
| current_object_cluster_fusion_v3 / F2_family_medoid_majority_guarded | 0.425721232 | 2951 | 3044 |
| current_object_cluster_fusion_v3 / F3_native_tie_original_four_vote | 0.431866645 | 2920 | 3013 |
| current_object_cluster_fusion_v3 / F4_medoid_quality_weighted_vote | 0.427702690 | 2940 | 3033 |
| current_proposal_assembly_v3 / C1_geo_raw_boundary_carve | 0.430938744 | 2958 | 3011 |
| current_proposal_assembly_v3 / C2_family_raw_boundary_carve | 0.430149968 | 2985 | 3011 |
| current_proposal_assembly_v3 / C3_geo_weak_bridge_reassemble | 0.430852049 | 2961 | 3010 |
| current_proposal_assembly_v3 / C4_family_bounded_pair_boundary | 0.431094683 | 2951 | 3012 |
| current_native_contour_v3 / C1_band2_weak035_core08 | 0.433275632 | 2910 | 3003 |
| current_native_contour_v3 / C2_band4_weak02_core065 | 0.433498312 | 2906 | 2999 |
| current_native_contour_v3 / C3_band4_agreement05 | 0.431512757 | 2922 | 3015 |
| current_native_contour_v3 / C4_opposite_branch_levelset_selection | 0.432276278 | 2917 | 3010 |
| current_composite_contour_v3 / C4_then_C2 | 0.436752508 | 2891 | 2955 |

## 다음 실험을 위한 원영상 특징

현재cloud에는707장원본JPEG가없고추론캐시만있다. 원본은사용자Drive에보존돼있다.
사진전체를다시전송하지않고다음특징만계산하는CPU전용셀을준비했다.

- 입력: 원본Ha사진과현재캐시. 정답JSON·모델·GPU는사용하지않는다.
- 특징: limb보정후multiscale dark-ridge Hessian(σ1/2/4),방향일관성,
  마스크내부와7px주변의밝기·median/MAD 대비. 주변에서다른기준예측전경은제외한다.
- 원반가장자리14px의ridge특징은유효영역밖으로표시하고,누락·빈주변을0으로꾸미지않는다.
- 대상: 기준마스크4,978개와원래foreground밖novel raw후보29,747개. 위치는예측만으로선택했다.
- 출력: RLEhash로연결되는수치CSV/JSONL과출처metadata. 사진과가중치는ZIP에담지않는다.
- 이미지단위원자적캐시로재개하며원본이미지·코드·파라미터변경시섞지않는다.
- 기능테스트9개와실제707장metadata/RLE preflight통과. 전체707장특징산출·실제수행시간·
  출력용량·PQ효과는아직미측정이다.

`SolarFilament_PixelQuality_CopyCell.py` 전체를Colab CPU런타임의새셀에복사한다.
Drive기본경로는기존과같고출력은 `fivefold_study_v2/pixel_quality_exports/v1` 이다.
완료후생성된특징ZIP을모두보내면원영상의선형구조가추가후보의FP를구분하는지시험할수있다.

새후보는개발비교의다음기준으로별도보존했다. 기존실제제출파이프라인과가중치는수정하지않았다.
`oof707_quality_v3_validation_only.csv`는검증예측이며대회test제출파일이아니다.
