# Paths for your machine. Everything else in this folder is relative to it. Source it (generate.sh does).
# --------------------------------------------------------------------------------------------------------
# Waymo Open Dataset processed by OpenPCDet (waymo_processed_data_v0_5_0/, ImageSets/, per-segment infos)
export WAYMO_ROOT=${WAYMO_ROOT:-/path/to/Waymo}
# OpenPCDet checkout (pcdet importable from here). Data loading only; no model is run during generation.
export OPENPCDET=${OPENPCDET:-/path/to/OpenPCDet}
# Where datasets are written: $OUT_ROOT/<dataset_name>/{dataset,annos,metrics}
export OUT_ROOT=${OUT_ROOT:-/path/to/atlas_datasets}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Trace libraries: the paper's ORIGINAL pickles ship in data/ (traces_{easy,medium,hard}.pkl, traces_val_cyclist_hard.pkl)
export TRACE_DIR=${TRACE_DIR:-$HERE/data}
# Paste object banks (too large for git): build with banks/build_val_object_bank.sh (README step 0)
export BANK_DIR=${BANK_DIR:-$OUT_ROOT/_banks/val_object_bank}     # vehicle/ and cyclist/object_bank_val.pkl
# --------------------------------------------------------------------------------------------------------
export GEN_DIR="$HERE/generators" DATA_DIR="$HERE/data"
export CP_SF_CFG=$OPENPCDET/tools/cfgs/waymo_models/centerpoint.yaml          # single-frame loader (scene points)
export CP_MF_CFG=$OPENPCDET/tools/cfgs/waymo_models/centerpoint_4frames.yaml  # 4-frame loader (frame indexing)
export CP_SF_CFG_RM=$OPENPCDET/tools/cfgs/custom_models/centerpoint_singleframe_waymo.yaml   # removal generator
export CP_MF_CFG_RM=$OPENPCDET/tools/cfgs/custom_models/centerpoint_multiframe_waymo.yaml    # removal generator
# ---- Slurm (submit_new.sh). Set for your cluster (NCSA Delta: CPU partition, your allocation's account) ----
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-CHANGE_ME}
export SLURM_PARTITION=${SLURM_PARTITION:-cpu}
export CONDA_ENV=${CONDA_ENV:-}      # env with pcdet; activated in each job if set (else the submitting env is inherited)
