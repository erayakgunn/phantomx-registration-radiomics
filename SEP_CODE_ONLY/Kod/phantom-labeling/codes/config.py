"""Configuration constants for CT Phantom Analysis Pipeline."""

import os

# Data source and output root. These are only used by the legacy
# ``main.py --data-dir/--output-dir`` CLI — the ``label_series.py``
# front end always takes paths on the command line. The defaults below
# point at the current user's Desktop; override them by exporting the
# ``CALIBRAMICS_DATA_ROOT`` / ``CALIBRAMICS_OUTPUT_ROOT`` environment
# variables or by passing ``--data-dir`` / ``--output-dir``.
_DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
DATA_ROOT = os.environ.get(
    "CALIBRAMICS_DATA_ROOT",
    os.path.join(_DESKTOP, "CT_Scans"),
)
OUTPUT_ROOT = os.environ.get(
    "CALIBRAMICS_OUTPUT_ROOT",
    os.path.join(_DESKTOP, "Pipeline_Output"),
)

# Phantom physical dimensions (cm)
PHANTOM_OUTER_DIAMETER_CM = 19.0
PHANTOM_OUTER_RADIUS_CM = PHANTOM_OUTER_DIAMETER_CM / 2.0

# Insert ring distances from center (cm) - measured from phantom images
OUTER_RING_DISTANCE_CM = 7.0
INNER_RING_DISTANCE_CM = 4.5

# ROI scale factor: extract 70% of detected pattern area to avoid edges.
# NOTE: Applies to Star and Texture ROIs only. Insert ROIs bypass this
# scale factor and use the full nominal insert radius instead (see below).
ROI_SCALE = 0.70

# Insert ROI diameter (cm) — VISUAL bright-core diameter, NOT the
# physical cylinder size. All 40 inserts are physically identical
# cylinders, so every insert ROI uses exactly this diameter with no
# per-ROI adaptive sizing.
#
# Calibration: radial profile of 120 inserts across 6 slices and 2
# sections at 0.586 mm/px gave:
#   * FWHM (50%-of-peak)  median radius = 17.09 px (~1.00 cm) — this is
#     the physical cylinder edge but it lies at the half-max of the
#     blurred edge, so a circle drawn here visibly bleeds into the dark
#     halo.
#   * 95%-of-peak min     radius = 14.21 px (~0.83 cm) — the worst-case
#     "still inside the visibly bright core" radius across all sampled
#     inserts.
# Drawn circle radius = ceil(INSERT_ROI_RADIUS_CM * px/cm) so we pick
# 0.75 cm = 12.80 px → ceil = 13 px, leaving ~1.2 px safety margin
# from the 14.21 px worst case. The circle is GUARANTEED to stay inside
# the visibly bright area of every insert.
INSERT_ROI_DIAMETER_CM = 1.5
INSERT_ROI_RADIUS_CM = INSERT_ROI_DIAMETER_CM / 2.0

# Texture ROI (cm)
TEXTURE_SIDE_CM = 3.9
TEXTURE_CENTER_DISTANCE_CM = 6.5


# Star pattern diameter (cm)
STAR_DIAMETER_CM = 4.0
STAR_RADIUS_CM = STAR_DIAMETER_CM / 2.0

# The 4 texture squares are one rigid module CONCENTRIC with the central
# star, so the star's radial-symmetry centre IS the texture-array centre.
# It is detected robustly on every protocol (strong spoke-convergence
# feature), whereas the phantom WALL centre from the boundary track is
# localized ~3-4 px off the feature array on soft-kernel / large-FOV recons
# (where the thin shell defeats contour detection). So the textures are
# anchored to the refined star centre, not the wall centre — making texture
# placement protocol-independent. This gate is the max distance (mm) the
# refined star centre may sit from the wall centre before it's distrusted
# (bad star fit) and the code falls back to the wall-centred texture mean.
STAR_ANCHOR_MAX_OFFSET_MM = 6.0

# HU thresholds
AIR_TISSUE_THRESHOLD_HU = -200

# Air-cavity discriminator — rejects solid anthropomorphic / body phantom
# sections that share the scan with the calibration phantom (these SEP
# acquisitions consistently include a soft-tissue body phantom at the
# high-z end of the series). The CalibraMics / PhantomX calibration phantom
# is air-cavity: even its densest insert slice keeps >75 % of the interior
# below -100 HU (air + thin tissue-equivalent plastic structures), and the
# rare dense honeycomb STAR8 slice still reads ~0.84. A solid body section
# has <6 % of its interior below -100 HU (mean interior HU ≈ +50). Gating
# STAR/INSERT eligibility on this fraction rejects the body REGARDLESS of how
# well the phantom circle was localized — critical because soft kernels
# (Br36f) and large FOV (500 mm) reconstructions blur/sub-pixel the thin
# shell so Phase 1 detection fails and falls back to the volume center,
# which would otherwise sample the (roughly centered) body and mislabel it
# as STAR16/STAR8. 0.40 sits ~0.35 clear of both classes (body ~0.05,
# phantom ≥0.78). Measured over the inner 0.7·R disc to stay clear of the
# wall/PVE annulus.
PHANTOM_INTERIOR_LOW_HU_MAX = -100.0
PHANTOM_MIN_LOW_HU_FRACTION = 0.40
PHANTOM_INTERIOR_SAMPLE_FRACTION = 0.70

# Per-slice boundary track (Phase 1) — see core/phantom_detector.py.
#
# The phantom centre is localized PER SLICE (primary: thin-shell contour;
# fallback: wall-independent interior-feature cluster), then the per-slice
# raw detections are turned into a smoothed (cx,cy,r) TRACK — a robust
# Theil-Sen line per axis — so every slice gets a data-driven boundary that
# follows real couch/gantry drift while rejecting per-slice detection noise.
# This replaces the old "collapse all slices to one global median (or
# volume-centre on failure)" behaviour.
#
# Outlier tolerance: a raw per-slice centre further than this from the
# series median is dropped before the line fit (gross misdetection, e.g. a
# stray cluster fit). ~25 px ≈ 1.5 cm at 0.586 mm/px — well beyond real
# drift (~8-9 px over a whole NAEOTOM series) but inside the ~2.5 cm
# inner/outer ring gap, so it never confuses phantom features.
PHANTOM_TRACK_OUTLIER_TOL_PX = 25.0

# Phantom detection (Hough)
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 200
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 40
HOUGH_MIN_RADIUS_CM = 8.0
HOUGH_MAX_RADIUS_CM = 11.0

# Minimum number of per-slice geometric phantom-circle matches (Phase 1)
# required before the median of those matches is trusted as the global
# center/radius. The required count is ``max(ABSOLUTE, FRACTION * num_slices)``
# — an absolute floor alone doesn't scale: 6/232 valid slices (2.6 %) on a
# real Siemens NAEOTOM Alpha PCCT series turned out just as unreliable as
# the SEP failure case this was written for, while 85/373 (23 %) on a SEP
# series is genuinely trustworthy. Below the required count, detection is
# treated as having effectively failed for the whole series and every
# slice falls back to the safe volume-center default instead. Guards
# against the case where only a handful of slices pass the contour gates —
# trusting their median would be no better than noise, and would also
# strand every other slice as INVALID instead of recovering via the
# fallback that already exists for the all-zero case.
PHANTOM_MIN_VALID_SLICES_FOR_TRUST = 5
PHANTOM_MIN_VALID_FRACTION_FOR_TRUST = 0.05

# Slice consistency
# Siemens PCCT scans (NAEOTOM Alpha) routinely produce ~8-9 px of total
# center drift across 360 slices of a 19 cm phantom (~5 mm lateral across
# ~36 cm scan length). The drift is a couch/gantry bias, not a detection
# error. 10 px matches observed Siemens behaviour while still catching
# real phantom slippage; previously 5 px fired on every NAEOTOM series.
MAX_CENTER_DRIFT_PX = 10
SLICE_SPACING_TOLERANCE = 0.10  # +/-10%

# Texture-validation edge trim (slices)
# Skip this many slices at each edge of a star section when emitting
# texture-score warnings. The physical texture inserts have non-zero
# z-extent and their first/last sampled slice captures a partial pattern
# (slice-averaged HU still includes background). These "partial pattern"
# slices aren't placement errors and shouldn't spam the validation log.
TEXTURE_VALIDATION_EDGE_TRIM = 3

# Section classification
MIN_SECTION_SLICES = 5  # Minimum slices for a valid section
SECTION_BOUNDARY_TRIM = 6  # Slices to mark as TRANSITION at each section boundary

# Max TRANSITION/INVALID gap (slices) that ``_find_star_groups`` and
# ``_find_insert_groups`` will bridge when grouping raw per-slice
# STAR/STAR8/STAR16/INSERT labels into the 2 physical modules of each kind.
# Per-slice Phase 1 contour misses (thin-shell phantoms, see the
# phantom-detector notes above) routinely punch 4-16 slice gaps WITHIN a
# single physical module — wider than the old value of 3, which fragmented
# one STAR16 + one STAR8 module into up to 5 separate runs (and one
# INSERT module into 2) and left every leftover run (everything but the
# top-2 by size) keeping its noisy raw per-slice label instead of being
# canonicalised — producing interleaved STAR16/STAR8 output, and a 3rd
# "INSERT" cluster that never got renamed INSERT1/INSERT2 and was silently
# dropped from extraction. 16 bridges the observed intra-module gaps while
# staying well under a real module's slice count, so it only merges
# fragments of the SAME module — genuinely different content (a different
# section type, a transition-zone joint) is labelled differently per-slice
# and never enters this gap-counting in the first place.
# ``_find_star_split`` (already designed for exactly this) then re-divides
# a merged star run into its STAR16 part and STAR8 part; insert groups
# need no such sub-split since there's only one raw "INSERT" label.
SECTION_RUN_MERGE_GAP = 16

# Inner-window trim — see ``core/section_trim.py``.
#
# Two mutually-exclusive ways to drive the trim:
#
#   * ``SECTION_INNER_WINDOW_HALF_MM`` — fixed half-window in mm. When set
#     to a positive number, every section keeps a ±N mm slab centred on
#     its detected z-midpoint, regardless of slice thickness. This is the
#     only way to get a TRULY protocol-independent labeled volume: a
#     ±10 mm window gives 20 slices on a 1 mm scan, 10 on a 2 mm scan,
#     7 on a 3 mm scan (ratio = 1:0.5:0.33, matching dz). Recommended
#     for cross-protocol comparison studies. Set to ``None`` to disable.
#
#   * ``SECTION_INNER_WINDOW_FRACTION`` — fallback when the half-mm is
#     ``None``. Keeps the inner fraction (default 0.7 = 70 %) of each
#     section's detected non-TRANSITION extent. NOT protocol-independent
#     because the detected extent itself varies with TRANSITION
#     sensitivity (thick-slice acquisitions over-trim and end up with
#     proportionally fewer kept slices). Use only when phantom module
#     dimensions are unknown and rough trimming is good enough.
#
# The fixed-mm option takes precedence when both are set.
SECTION_INNER_WINDOW_HALF_MM = 15.0
SECTION_INNER_WINDOW_FRACTION = 0.7

# STAR8/STAR16 adaptive PVE guard (see ``core/roi_extractor.py``,
# ``_extract_star_rois``) — after the fixed-mm inner-window trim above has
# already picked the section's slices, this guard peels ADDITIONAL leading/
# trailing slices whose gradient signal indicates a not-yet-stable shape
# (star spokes / texture squares still forming or fading into the module
# boundary). The peel budget must be expressed in physical mm, not a raw
# slice count: a fixed slice-count cap (e.g. "3 slices") is 3 mm of real
# margin on a 1 mm-thick protocol but 9 mm on a 3 mm-thick protocol, so the
# same cap either under-protects thin-slice scans or over-trims (and can
# clip correctly-formed slices from) thick-slice scans. Expressing the
# budget in mm keeps the guard's physical reach constant across protocols,
# matching ``SECTION_INNER_WINDOW_HALF_MM``'s protocol-independence.
STAR_ADAPTIVE_GUARD_MAX_MM = 3.0

# Physical safety buffer (mm) the adaptive guard enforces between a kept
# STAR window edge and the module-edge signal COLLAPSE (slice whose
# star-disc gradient falls below 0.75 × the section median — real
# transition material, unreachable by plateau noise). A kept edge slice
# closer than this to a collapse is peeled (and the window slid to hold
# the target count). Expressed in mm for protocol independence: 2 mm is
# 2 slices at dz=1 mm, 1 slice at dz≥2 mm. Chosen from verified cases:
# KSW STAR8 slice 107 sat 2 slices from its collapse and was visually in
# the transition (must peel → needs buffer ≥ 2 at dz=1), while KSW
# low-dose SharpPlus STAR16 slice 51 sat 6+ slices clear and was clean
# (must keep — any buffer ≤ 5 works). See _extract_star_rois.
STAR_GUARD_CLIFF_BUFFER_MM = 2.0

# Insert detection
INSERTS_PER_RING = 10
INSERTS_PER_SECTION = 20  # 10 outer + 10 inner

# Canonical slot-0 angle (degrees, image coords: 0°=right, 90°=down) for each
# insert ring. Used to ANCHOR the numbering so slot 0 is always the SAME
# physical insert across every protocol. The OUTER ring's slot 0 (Insert_4)
# sits at 0° (an insert at 3 o'clock); the INNER ring is staggered by half a
# slot, so its slot 0 (Insert_15) sits at 18°. These are physical geometry of
# the (single) CalibraMics phantom, not protocol tuning.
#
# Why this matters: the start-angle fit returns a phase in [0, 36°), which is
# 36°-periodic and so cannot say WHICH insert is slot 0. The outer ring's
# slot 0 lies exactly on that 0/36° boundary, so detection noise that nudges
# the fitted phase from +0.3° to 35.8° (the SAME phase) silently rotates the
# whole outer ring by one slot — the "numbering shifted one to the right on
# some protocols" symptom (seen on KSW; SEP/PCCT happened to land on the safe
# side). Snapping the fit to the grid representative nearest these reference
# angles pins slot 0 to a fixed physical insert. The inner ring's reference
# (18°) is mid-interval so it was never affected, which is why only the outer
# ring flipped.
INSERT_OUTER_SLOT0_ANGLE_DEG = 0.0
INSERT_INNER_SLOT0_ANGLE_DEG = 18.0

# Per-slice insert refinement: search windows tried in order around each
# expected insert position. Diameter (not radius). The 2 cm window was
# DROPPED — a 1.5 cm insert is ~58 % of a 2 cm window, so the partial-
# volume rim on the far side from the seed gets clipped by the window
# boundary, biasing the Stage 1 centroid by 1–3 px toward the seed.
# 3 cm gives the insert plenty of headroom (insert is ~25 % of window
# area) for an unbiased centroid; 4 cm fallback handles larger seed
# errors. Stage 2 Hough still uses a tight 2 cm window centred on the
# (now unbiased) Stage 1 anchor for sub-pixel precision.
INSERT_SEARCH_WINDOW_DIAMETERS_CM = (3.0, 4.0)

# Fallback HU floor used by ``refine_insert_center`` only when Otsu cannot
# produce a usable bimodal split (e.g. windows with <30 HU dynamic range).
# The primary thresholding path is adaptive (Otsu inside the window with
# minority-class selection), which handles BOTH air-cavity (Siemens) and
# solid-plastic (Philips) phantoms — the insert is whichever side of the
# Otsu split covers <50% of the search window.
INSERT_NON_AIR_THRESHOLD_HU = -500

# Refined centre must land within this many pixels of the (data-derived,
# per-ring) expected radius; otherwise the refinement is treated as failed
# and Stage C falls back to the ring formula. ~5 px ≈ 3 mm at 0.586 mm/px,
# well below the ~2.5 cm radial gap between outer and inner rings.
INSERT_REFINEMENT_RADIUS_TOL_PX = 5.0

# Tolerance used to scrub the detect_inserts cloud before fitting the
# start angle. Phantom wall artefacts and mount features that the legacy
# Hough/CC detector misclassifies as "outer ring" sit ~2 cm beyond the
# real outer ring; filtering at ±15 px (~0.9 cm) drops them without
# excluding real inserts on phantoms whose ring radius differs slightly
# from the config nominal. This was the root cause of outer-ring labels
# being shifted relative to inner-ring labels (inner ring sits in a
# cleaner detection neighbourhood — no nearby wall to pollute its cloud).
INSERT_RING_RADIUS_FILTER_TOL_PX = 15.0

# Minimum number of successful per-slice refinements required before
# Stage C trusts the per-section median anchor. Below this, the insert
# falls back to the ring-formula seed for that section.
INSERT_REFINEMENT_MIN_SLICES = 2

# Mean-image polish (Stage C2) max shift, in mm, for the TWO insert classes:
#   * a "good-anchor" insert (≥INSERT_REFINEMENT_MIN_SLICES per-slice
#     detections, so its anchor is the trustworthy per-section median) may
#     only be polished a tiny amount — a large shift there would be the mean
#     image mis-locking, so cap it tight (3 mm).
#   * a "fallback" insert (too faint to refine per-slice on soft-kernel /
#     large-FOV recons, so its anchor is the imperfect ring FORMULA) often
#     needs to move ~1 cm to snap onto the real insert that IS clearly
#     visible in the high-SNR mean image. The tight 3 mm cap was rejecting
#     exactly that correction, leaving these inserts (e.g. Insert_06 on the
#     500 mm-FOV Br36f series) stranded on the insert's rim. 12 mm is enough
#     to reach the real insert while staying well under the inter-insert
#     spacing (inner ring chord ≈ 16 mm, outer ≈ 25 mm) so it can never snap
#     onto a neighbour.
INSERT_POLISH_MAX_SHIFT_MM = 3.0
INSERT_FALLBACK_POLISH_MAX_SHIFT_MM = 12.0

# When iterative slot calibration finds a start-angle shift larger than
# this (degrees), the loop re-seeds initial positions and re-runs Stage B
# once. Keeps Insert_NN labels glued to the right physical slot when the
# initial detect_inserts cloud was noisy.
INSERT_SLOT_CALIBRATION_REFIT_DEG = 2.0

# Per-slice ROI extraction mode.
#
# Default False (shared per-section anchor): every slice in a section gets
# its ROI extracted at the SAME (cx, cy) — the per-section median of all
# per-slice refinements after mean-image polish. This is the right answer
# for a RIGID PHANTOM: the physical insert centre is identical across all
# slices in a section, so the consensus average IS the truth. Per-slice
# detection noise (random pixel noise, PVE rim variation, beam-hardening
# artefacts) is averaged out, producing zero visible fluctuation between
# consecutive slices in the painted overlays.
#
# True (dynamic per-slice): each slice's ROI uses its own refined centre
# from Stage B2. The painted overlays will show 0.3–1 px fluctuations
# between consecutive slices — that's the actual per-slice refinement
# variance, not a bug. Only useful when you NEED to verify per-slice
# detection visually, or when the phantom is suspected of z-dependent
# drift that the shared anchor would average away.
#
# Per-slice detection (Stages B and B2) runs in BOTH modes; the flag only
# controls whether Stage D consumes per-slice positions or the consensus
# anchor.
INSERT_ROI_PER_SLICE_DYNAMIC = False

# Sanity tolerance for per-slice ROI centres in dynamic mode. A per-slice
# refinement that lands further than this from the per-section anchor is
# treated as a refinement failure and that slice falls back to the anchor.
INSERT_ROI_PER_SLICE_MAX_DEV_MM = 3.0

# Concentric sub-ROI diameters drawn from each refined insert centre.
# Each insert produces one ROI (.dcm + overlay) per listed diameter.
#
# 2026-09-15: default reduced from (5, 10, 15) to (5, 10). The 15 mm crop
# was never consumed by the radiomics extraction (``INSERT_SIZES`` there is
# 5/10 mm only) and the USZ 15 mm patches were deleted on 2026-09-06; it
# only cost a third correlated row per insert plus export time. Pass
# ``--insert-sizes 5,10,15`` to ``label_series.py`` / ``main.py`` to
# reproduce the old three-size layout.
INSERT_SUBROI_DIAMETERS_MM = (5, 10)

# Overlay colours (BGR for OpenCV) keyed by sub-ROI diameter in mm.
# The 15 mm entry is kept so that ``--insert-sizes 5,10,15`` still paints.
INSERT_SUBROI_COLORS_BGR = {
    5:  (0, 255, 255),    # yellow
    10: (255, 255, 0),    # cyan
    15: (255, 0, 255),    # magenta (only with --insert-sizes 5,10,15)
}

# Minimum angular modulation depth before a slice is called a star at all
# (``analysis/star_classifier.classify_star``). Same quantity and same scale
# as ``core/validator.STAR_MIN_MODULATION``: the HU swing of the arms at the
# winning rotational order, over the sampled ring's own 5-95 percentile
# contrast.
#
# Measured on the 336-series PCCT run: genuine star slices sit at 0.38-0.58
# (STAR16 lower than STAR8 because 16 narrower arms are blurred more by the
# reconstruction); insert slices, inter-module transitions and plain plastic
# sit below 0.06. 0.15 sits in that gap.
#
# The classifier is deliberately STRICTER about what it will call a star
# than the validator is about confirming one, because a wrong star label
# here moves a section boundary and changes which slices get exported.
STAR_CLASSIFY_MIN_MODULATION = 0.15

# Star classification FFT (legacy peak-count ranges — no longer consulted by
# ``classify_star``, which now counts arms directly in polar space. Kept for
# the diagnostic scripts that still report peak counts.)
STAR16_PEAK_RANGE = (14, 20)
STAR16_PEAK_RANGE_HIGH = (28, 44)
STAR8_PEAK_RANGE = (5, 11)

# Texture validation
#
# Legacy gate for ``_fit_sine_r2``. No longer drives the Sinusoidal card
# verdict (see SINE_HARMONIC_PURITY_MIN); kept because the helper is still
# available for diagnostics.
SINE_FIT_R2_THRESHOLD = 0.7

# How far the lattice ring must stand above the rest of the patch's radial
# spectrum before a texture card counts as carrying a real pattern
# (``analysis/texture_validator._lattice_ring_order``).
#
# Purpose: catch the failure that actually matters — an ROI that slid off
# its card and is sampling featureless plastic. Noise has no preferred
# spatial frequency, so its radial spectrum is flat and its ring
# prominence sits just above 1; a real lattice puts a sharp ring wherever
# its pitch lands.
#
# Measured: sampled texture patches across all four PCCT studies (every
# card type, 80-120 kVp, 300-500 mm FOV, 1 and 3 mm recon) bottomed out at
# 3.9 (Sinusoidal, the shallowest-contrast card); synthetic featureless
# plastic scored 1.22-1.36. 2.5 sits in that gap. The margin is modest
# because it is a RATIO of spectral means — the separation in the
# underlying quantity spans three orders of magnitude (see the function's
# note), so a patch anywhere near this floor is genuinely pattern-free.
LATTICE_MIN_RING_PROMINENCE = 2.5

# Minimum harmonic purity (1 - |3f|/|f|) for the Sinusoidal card.
#
# Separates the phantom's sine card from the square-wave bar gratings: a
# sine has no third harmonic, an ideal square wave has one at 1/3 the
# fundamental amplitude. Read straight off the spectrum, so unlike the old
# ``curve_fit`` R² gate there is no optimiser to converge badly — that gate
# produced all 21 "Sinusoidal sine_fit low" warnings of the 336-series run
# on patches physically indistinguishable from the passing ones (see
# ``_validate_sinusoidal``).
#
# Measured: real Sinusoidal patches scored ≥ 0.947 (warned ones 0.971-0.974);
# a synthetic square-wave bar grating scores 0.587. 0.75 sits in that gap.
SINE_HARMONIC_PURITY_MIN = 0.75

# Visualization
OVERLAY_SIZE = (512, 512)
WINDOW_WIDTH = 400
WINDOW_LEVEL = 0

# Section definitions
# Texture angular positions: 0=top(270deg image), 90=right(0deg image),
# 180=bottom(90deg image), 270=left(180deg image)
# In image coordinates: top=-90deg, right=0deg, bottom=90deg, left=180deg
SECTIONS = {
    "STAR16": {
        "star_arms": 16,
        "textures": {
            270: "Diagonal",     # top in image
            0: "Vertical",       # right in image
            90: "Square",        # bottom in image
            180: "Horizontal",   # left in image
        },
    },
    "STAR8": {
        "star_arms": 8,
        "textures": {
            270: "Honeylarge",   # top in image
            0: "Honeymedium",    # right in image
            90: "Sinusoidal",    # bottom in image
            180: "Honeysmall",   # left in image
        },
    },
    # Outer/inner insert lists: index = slot number, value = Insert_NN
    # that physically sits at that slot in the reference image.
    #
    # With ``canonicalize_orientation`` ensuring every scan is presented
    # in the same canonical (stars-first, non-mirrored) form, the slot-0
    # fit consistently lands at ≈0° (image right). On the outer ring:
    #   slot 0  (≈  0°, right) → Insert_4
    #   slot 7  (≈252°, upper-left of top pair) → Insert_1
    #   slot 8  (≈288°, upper-right of top pair) → Insert_2
    # which matches the user's hand-labeled reference (1 and 2 straddle
    # the top, then 3..10 continue clockwise).
    #
    # On the inner ring the slot-0 fit lands at ≈18° (right side, just
    # below horizontal):
    #   slot 0  (≈ 18°) → Insert_15
    #   slot 6  (≈234°, upper-left of inner) → Insert_11
    #   slot 7  (≈270°, top of inner) → Insert_12
    #
    # If a future scan shows the labels off by one, cyclic-shift the
    # affected list — the entire ring rotates together since the numbering
    # is strictly clockwise.
    "INSERT1": {
        "outer_inserts": [4, 5, 6, 7, 8, 9, 10, 1, 2, 3],
        "inner_inserts": [15, 16, 17, 18, 19, 20, 11, 12, 13, 14],
    },
    "INSERT2": {
        "outer_inserts": [24, 25, 26, 27, 28, 29, 30, 21, 22, 23],
        "inner_inserts": [35, 36, 37, 38, 39, 40, 31, 32, 33, 34],
    },
}

# Section order along z-axis (ascending slice location)
SECTION_ORDER = ["STAR16", "STAR8", "INSERT1", "INSERT2"]
