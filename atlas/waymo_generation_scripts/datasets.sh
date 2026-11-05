# Registry of every 50-segment ATLAS Waymo dataset (CVPR'27). Source after config.sh.
#   spec <name>  -> sets GEN (global|relative|removal|baseline), ARGS (array), IMODE, JIT_IN, JIT_INTER, NOTE
# All datasets: seed 42, the 50 val segments in data/segments_50_seed42.txt (drawn by the generators with
# --num_sequences 50 --sequence_selection random --seed 42 from the 195 segments common to the 1- and 4-frame
# loaders), alternating 32-frame clean/attack blocks (clean 0-31, attack 32-63, ...).
ATT_I=0.3221      # attenuated intensity: q90 of real-car trace intensity, tanh scale, pooled over the 3 libraries
SPLITS=(global_easy global_medium global_hard rel_fixed_easy rel_fixed_medium rel_fixed_hard)

spec () {
    local n=$1 k
    GEN=global; ARGS=(); IMODE=adaptive; JIT_IN=0; JIT_INTER=0; VIEW=0; NOTE=""
    local FULL32=(--continuous_min_len 32 --continuous_max_len 32)
    local SEL=(--num_sequences ${NSEQ:-50} --sequence_selection ${SELECTION:-random})   # NSEQ=1 SELECTION=first: smoke test
    local CAR_TR=$TRACE_DIR/traces_hard.pkl CYC_TR=$TRACE_DIR/traces_val_cyclist_hard.pkl
    local CARP=(--trace_file $CAR_TR --spoof_source gt_paste --object_bank $BANK_DIR/vehicle/object_bank_val.pkl --target_class Vehicle)
    local CYCP=(--trace_file $CYC_TR --spoof_source gt_paste --object_bank $BANK_DIR/cyclist/object_bank_val.pkl --target_class Cyclist)
    case "$n" in
      # ---- main point injection: <placement>_<budget>_seed42 (adaptive) / _att_seed42 (attenuated) ----
      global_easy_seed42|global_medium_seed42|global_hard_seed42|global_easy_att_seed42|global_medium_att_seed42|global_hard_att_seed42)
          k=${n#global_}; k=${k%%_*}; ARGS=(--trace_file $TRACE_DIR/traces_$k.pkl "${FULL32[@]}" "${SEL[@]}")
          if [[ $n == *_att_* ]]; then IMODE=attenuated; fi ;;
      rel_fixed_easy_seed42|rel_fixed_medium_seed42|rel_fixed_hard_seed42|rel_fixed_easy_att_seed42|rel_fixed_medium_att_seed42|rel_fixed_hard_att_seed42)
          GEN=relative; k=${n#rel_fixed_}; k=${k%%_*}; ARGS=(--trace_file $TRACE_DIR/traces_$k.pkl "${SEL[@]}")
          if [[ $n == *_att_* ]]; then IMODE=attenuated; fi ;;
      # ---- intensity: the trace's own recorded intensities ----
      *_trace_seed42)
          spec "${n%_trace_seed42}_seed42"; IMODE=trace; return ;;
      # ---- attenuated + ONE spoofer-jitter component (Sato NDSS'24, along each laser ray) ----
      *_att_jpt_50_seed42)   # per point: N(0, 0.10 m) per point, object does not move as a whole
          spec "${n%_att_jpt_50_seed42}_att_seed42"; JIT_IN=0.10; JIT_INTER=0; return ;;
      *_att_jfr_50_seed42)   # per frame: one N(0, 0.35 m) shift per frame for the whole object
          spec "${n%_att_jfr_50_seed42}_att_seed42"; JIT_IN=0; JIT_INTER=0.35; return ;;
      # ---- post-rebuttal ablations (global generator, ablation switches) ----
      abl_car_atlas_hard_global_seed42)  ARGS=(--trace_file $CAR_TR "${FULL32[@]}" "${SEL[@]}") ;;
      abl_car_atlas_hard_flicker_seed42) ARGS=(--trace_file $CAR_TR "${FULL32[@]}" "${SEL[@]}" --random 0.5) ;;
      abl_car_atlas_hard_randobj_seed42) ARGS=(--trace_file $CAR_TR "${FULL32[@]}" "${SEL[@]}" --object_mode per_frame) ;;
      abl_cyc_atlas_global_seed42)       ARGS=(--trace_file $CYC_TR --target_class Cyclist "${FULL32[@]}" "${SEL[@]}") ;;
      abl_cyc_atlas_relative_seed42)     ARGS=(--trace_file $CYC_TR --target_class Cyclist "${FULL32[@]}" "${SEL[@]}" --positioning relative) ;;
      abl_cyc_atlas_flicker_seed42)      ARGS=(--trace_file $CYC_TR --target_class Cyclist "${FULL32[@]}" "${SEL[@]}" --random 0.5) ;;
      abl_cyc_atlas_randobj_seed42)      ARGS=(--trace_file $CYC_TR --target_class Cyclist "${FULL32[@]}" "${SEL[@]}" --object_mode per_frame) ;;
      abl_*_atlas_*_att_seed42)
          spec "${n%_att_seed42}_seed42"; IMODE=attenuated; return ;;
      # paste = real object points + real intensities from the val bank (GT-sampling style); intensity mode n/a
      abl_car_paste_global_seed42)   ARGS=("${CARP[@]}" "${FULL32[@]}" "${SEL[@]}") ;;
      abl_car_paste_relative_seed42) ARGS=("${CARP[@]}" "${FULL32[@]}" "${SEL[@]}" --positioning relative) ;;
      abl_car_paste_flicker_seed42)  ARGS=("${CARP[@]}" "${FULL32[@]}" "${SEL[@]}" --random 0.5) ;;
      abl_car_paste_randobj_seed42)  ARGS=("${CARP[@]}" "${FULL32[@]}" "${SEL[@]}" --object_mode per_frame) ;;
      abl_cyc_paste_global_seed42)   ARGS=("${CYCP[@]}" "${FULL32[@]}" "${SEL[@]}") ;;
      abl_cyc_paste_relative_seed42) ARGS=("${CYCP[@]}" "${FULL32[@]}" "${SEL[@]}" --positioning relative) ;;
      abl_cyc_paste_flicker_seed42)  ARGS=("${CYCP[@]}" "${FULL32[@]}" "${SEL[@]}" --random 0.5) ;;
      abl_cyc_paste_randobj_seed42)  ARGS=("${CYCP[@]}" "${FULL32[@]}" "${SEL[@]}" --object_mode per_frame) ;;
      # ==== NEW (paper resubmission, Nov 2026): see NEW_DATASETS.md ====
      # ---- class-wise injection: cyclist easy/medium (hard = abl_cyc_atlas_{global,relative}), pedestrian all 3 ----
      abl_cyc_atlas_global_easy_seed42|abl_cyc_atlas_global_medium_seed42|abl_cyc_atlas_relative_easy_seed42|abl_cyc_atlas_relative_medium_seed42|abl_ped_atlas_global_easy_seed42|abl_ped_atlas_global_medium_seed42|abl_ped_atlas_global_hard_seed42|abl_ped_atlas_relative_easy_seed42|abl_ped_atlas_relative_medium_seed42|abl_ped_atlas_relative_hard_seed42)
          local c=${n#abl_}; c=${c%%_*}; local pl=${n#abl_${c}_atlas_}; pl=${pl%%_*}; local b=${n%_seed42}; b=${b##*_}
          local cls=Cyclist dir=cyclist3 low=cyclist; [[ $c == ped ]] && { cls=Pedestrian; dir=pedestrian; low=pedestrian; }
          ARGS=(--trace_file $BANK_DIR/$dir/traces_val_${low}_${b}.pkl --target_class $cls "${FULL32[@]}" "${SEL[@]}")
          [[ $pl == relative ]] && ARGS+=(--positioning relative)
          NOTE="needs banks/build_class_trace_banks.sh first" ;;
      # ---- viewpoint-consistent phantom: points re-sampled perpendicular to each ray every frame (0.05 m) ----
      global_medium_att_view_seed42|global_hard_att_view_seed42)
          spec "${n%_view_seed42}_seed42"; VIEW=0.05; return ;;
      # ---- scene-level perturbations (gen_scene_perturb.py); targets copied from atlas_removal_az20_seed42 ----
      spray_l1_seed42|spray_l2_seed42|crosstalk_006_seed42|crosstalk_010_seed42|fog_a002_seed42|fog_a006_seed42|blackout_3_seed42|blackout_10_seed42|stuck_4_seed42|stuck_10_seed42|delayed_1_seed42|delayed_3_seed42|dropped_025_seed42|dropped_050_seed42|pose_noise_low_seed42|pose_noise_high_seed42)
          GEN=scene; local op=${n%_seed42}; local lv=${op##*_}; op=${op%_*}; local P
          case $op in
            spray)      P="{\"level\": ${lv#l}, \"min_speed\": ${SPRAY_MIN_SPEED:-15.0}}" ;;
            crosstalk)  P="{\"frac\": 0.${lv}}" ;;
            fog)        P="{\"alpha\": 0.${lv#a00}}"; [[ $lv == a002 ]] && P='{"alpha": 0.02}'; [[ $lv == a006 ]] && P='{"alpha": 0.06}' ;;
            blackout)   P="{\"length\": $lv}" ;;
            stuck)      P="{\"repeats\": $lv}" ;;
            delayed)    P="{\"delay\": $lv}" ;;
            dropped)    P="{\"p\": 0.${lv#0}}" ;;
            pose_noise) [[ $lv == low ]] && P='{"rot_sigma": 0.04, "trans_sigma": 0.004}' || P='{"rot_sigma": 0.10, "trans_sigma": 0.010}' ;;
          esac
          ARGS=(--perturbation $op --params "$P" --reference $OUT_ROOT/${SCENE_REF:-atlas_removal_az20_seed42})
          NOTE="needs \$OUT_ROOT/${SCENE_REF:-atlas_removal_az20_seed42} (paired targets)" ;;
      # ---- adverse point-loss baselines (scene-wide, same frames/targets as ATLAS removal) ----
      baseline_uniform_50_seed42)   GEN=baseline; ARGS=(--mode uniform --drop_prob 0.50) ;;
      baseline_uniform_90_seed42)   GEN=baseline; ARGS=(--mode uniform --drop_prob 0.90) ;;
      baseline_ring_16_seed42)      GEN=baseline; ARGS=(--mode ring --n_rings 16 --inclination_file $DATA_DIR/waymo_top_lidar_inclinations.npy) ;;
      baseline_ring_32_seed42)      GEN=baseline; ARGS=(--mode ring --n_rings 32 --inclination_file $DATA_DIR/waymo_top_lidar_inclinations.npy) ;;
      baseline_matched_az20_seed42) GEN=baseline; ARGS=(--mode matched --az_width 20) ;;
      baseline_matched_az50_seed42) GEN=baseline; ARGS=(--mode matched --az_width 50) ;;
      # ---- ATLAS A-HFR point removal, paired across widths and with the baselines ----
      atlas_removal_az10_seed42|atlas_removal_az20_seed42|atlas_removal_az30_seed42|atlas_removal_az40_seed42|atlas_removal_az50_seed42|atlas_removal_az60_seed42)
          GEN=removal; k=${n#atlas_removal_az}; k=${k%_seed42}
          ARGS=(--az_width $k --attack_block_length 32 --desired_target_range_m 60.0 --tracking_max_step_m 6.0
                --target_class 1 --el_width_deg 16.0 --max_az_step_deg 15.0 --center_jitter_std_deg 0.05
                --min_target_range_m 6.0 --pairing_reference_metrics $OUT_ROOT/baseline_matched_az50_seed42/metrics
                --pairing_box_atol 1e-5)
          NOTE="needs baseline_matched_az50_seed42 first (pairing check)" ;;
      *) echo "unknown dataset: $n (see README for the list)" >&2; return 1 ;;
    esac
}

# every dataset name, in a sensible generation order (baselines before removal)
all_datasets () {
    local s v
    for v in "" _att; do for s in "${SPLITS[@]}"; do echo ${s}${v}_seed42; done; done
    for s in global_easy global_hard rel_fixed_easy rel_fixed_hard; do echo ${s}_trace_seed42; done
    for k in jpt jfr; do for s in "${SPLITS[@]}"; do echo ${s}_att_${k}_50_seed42; done; done
    for v in "" _att; do
        for a in global flicker randobj; do echo abl_car_atlas_hard_${a}${v}_seed42; done
        for a in global relative flicker randobj; do echo abl_cyc_atlas_${a}${v}_seed42; done
    done
    for c in car cyc; do for a in global relative flicker randobj; do echo abl_${c}_paste_${a}_seed42; done; done
    for b in uniform_50 uniform_90 ring_16 ring_32 matched_az20 matched_az50; do echo baseline_${b}_seed42; done
    for w in 10 20 30 40 50 60; do echo atlas_removal_az${w}_seed42; done
}

# the datasets added for the resubmission (NEW_DATASETS.md), in generation order
new_datasets () {
    local p b
    for p in global relative; do for b in easy medium; do echo abl_cyc_atlas_${p}_${b}_seed42 abl_cyc_atlas_${p}_${b}_att_seed42; done; done | tr ' ' '\n'
    for p in global relative; do for b in easy medium hard; do echo abl_ped_atlas_${p}_${b}_seed42 abl_ped_atlas_${p}_${b}_att_seed42; done; done | tr ' ' '\n'
    echo global_medium_att_view_seed42; echo global_hard_att_view_seed42
    for n in spray_l1 spray_l2 crosstalk_006 crosstalk_010 fog_a002 fog_a006 blackout_3 blackout_10 stuck_4 stuck_10 \
             delayed_1 delayed_3 dropped_025 dropped_050 pose_noise_low pose_noise_high; do echo ${n}_seed42; done
}
