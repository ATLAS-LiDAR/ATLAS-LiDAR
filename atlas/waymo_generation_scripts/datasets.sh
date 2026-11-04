# Registry of every 50-segment ATLAS Waymo dataset (CVPR'27). Source after config.sh.
#   spec <name>  -> sets GEN (global|relative|removal|baseline), ARGS (array), IMODE, JIT_IN, JIT_INTER, NOTE
# All datasets: seed 42, the 50 val segments in data/segments_50_seed42.txt (drawn by the generators with
# --num_sequences 50 --sequence_selection random --seed 42 from the 195 segments common to the 1- and 4-frame
# loaders), alternating 32-frame clean/attack blocks (clean 0-31, attack 32-63, ...).
ATT_I=0.3221      # attenuated intensity: q90 of real-car trace intensity, tanh scale, pooled over the 3 libraries
SPLITS=(global_easy global_medium global_hard rel_fixed_easy rel_fixed_medium rel_fixed_hard)

spec () {
    local n=$1 k
    GEN=global; ARGS=(); IMODE=adaptive; JIT_IN=0; JIT_INTER=0; NOTE=""
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
