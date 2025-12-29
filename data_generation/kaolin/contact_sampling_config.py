CONFIG = {
  "paths": {"static_mesh":"./data_generation/assets/meshes/cylinder_simple/cylinder_simple_hole.obj","moving_mesh":"./data_generation/assets/meshes/cylinder_simple/cylinder_simple_peg.obj","csv_out":"./data_generation/data/cylinder_simple/cylinder_simple_contact_poses.csv","npz_out":"./data_generation/data/cylinder_simple/cylinder_simple_contact_poses.npz"},
  "device":"cuda",
  "seed":0,
  "bounds":{"tmin":[-0.002,-0.002,0.0],"tmax":[0.002,0.002,0.020],"rmin":[-5,-5,-5],"rmax":[5,5,5]}, # rmin/rmax in degrees
    # ------------------------------------------------------------
    # Sampling and batching
    # ------------------------------------------------------------
    "sampling": {
        # Total number of pose samples (rows) to be written to CSV/NPZ.
        # This is the *global budget* for the adaptive sampler.
        # Coarse + fine samples both count toward this if enabled.
        "budget_rows": 10_000_000,

        # Number of poses evaluated in parallel on the GPU per kernel call.
        # Larger = better GPU utilization but higher VRAM usage.
        # Typically 512–8192 depending on mesh size and GPU memory.
        "gpu_batch": 2**14,

        # Number of probe poses sampled when evaluating a 6D cell
        # to decide whether it is free space or needs refinement.
        # Usually <= gpu_batch.
        # Higher = more reliable refinement decisions, slower exploration.
        "probe_per_cell": 1024,
    },

    # ------------------------------------------------------------
    # Distance / contact thresholds (units match mesh units, e.g., meters)
    # ------------------------------------------------------------
    "thresholds": {
        # If both |signed distance| and min separation exceed this value,
        # the region is considered "far free space" and will not be refined.
        # Controls how aggressively empty space is pruned.
        "far_thresh": 0.0025,        # e.g., 1 cm

        # Defines the "near-contact band":
        # poses with |signed distance| < near_thresh are treated as
        # potentially contacting and will trigger refinement.
        # Smaller values = sharper contact boundary, more refinement.
        "near_thresh": 0.002,      # e.g., 2 mm

        # Maximum allowed penetration depth for a pose to be considered
        # a valid (near-contact) configuration.
        # Penetrations deeper than this are labeled invalid.
        "epsilon_pen": 0.0005,     # e.g., 0.5 mm
    },

    # ------------------------------------------------------------
    # Surface sampling densities for distance approximation
    # ------------------------------------------------------------
    "surface_samples": {
        # Number of surface samples taken from the moving mesh
        # during coarse evaluation.
        # Used to approximate signed distance cheaply in free space.
        # Low values keep exploration fast.
        "moving_coarse": 2**10,

        # Number of surface samples taken from the moving mesh
        # during fine (near-contact) evaluation.
        # Higher values improve accuracy of separation/penetration estimates.
        "moving_fine": 2**14,

        # Number of surface samples taken from the static mesh
        # for symmetric distance and penetration checks (static → moving).
        # Used only in fine evaluation.
        "static_fine": 2**14,

        # Maximum number of static samples used during *coarse probing*.
        # This caps the cost of coarse checks while still detecting penetration.
        # Must be <= static_fine.
        "static_probe_cap": 2**10,
    },

    # ------------------------------------------------------------
    # Output / logging behavior
    # ------------------------------------------------------------
    "io": {
        # Number of rows accumulated in memory before flushing to CSV.
        # Larger values reduce disk I/O but increase RAM usage.
        "csv_flush_every": 8192*64,

        # Print progress every N written rows.
        # Useful for long runs to monitor convergence and performance.
        "print_every": 8192*8,

        # Whether to also save a compressed NPZ file at the end.
        # NPZ is convenient for fast loading in Python but requires RAM.
        "write_npz": True,
    },

    # ------------------------------------------------------------
    # Adaptive refinement behavior
    # ------------------------------------------------------------
    "refinement": {
        # If True, poses detected as very close to contact are re-evaluated
        # with higher surface sampling density for higher accuracy.
        "enable_fine_upgrade": True,

        # Multiplier applied to near_thresh to define "very near" contact.
        # A pose is considered very near if:
        #   |signed distance| < very_near_factor * near_thresh
        # Smaller values mean fewer fine upgrades, higher confidence.
        "very_near_factor": 0.5,

        # If True, fine-evaluated poses consume budget_rows.
        # If False, fine evaluations are "free" refinements and only
        # coarse samples count toward the global budget.
        "count_fine_toward_budget": True,
    },

}
