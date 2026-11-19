# Ventral (motor) selection mask


def filter_ventral_root_trajectory_rows(df, name_col="NeuronName"):
    """
    Select the same ventral-root trajectories as the original
    Sim4Life entity-selection code.

    Selection
    ---------
    T7-T12:
        - name contains:
            "_ventral_root_T7_"
            ...
            "_ventral_root_T12_"

    L1-L2:
        - name starts with:
            "Ventral_root_L1"
            or
            "Ventral_root_L2"

        - exclude names containing "cutaneous"

        - final trajectory index must be < 2
          (i.e. _01 in the current dataset)

    Rows must specifically belong to:

        Discretized_Nerves/
        Ventral_Root_Nerves/
        Ventral_Root_Nerves_(T7-L2)_XXum

    Expected:
        126 rows total
        = 42 trajectories × 3 fibre diameters
    """

    if name_col not in df.columns:
        raise KeyError(
            f"Could not find column '{name_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    names = df[name_col].astype(str)


    # ---------------------------------------------------------
    # Extract original Sim4Life trajectory/entity name
    # ---------------------------------------------------------
    #
    # Example:
    #
    # Nerve_ventral_root_T12_..._02
    # [Motor MRG Neuron 20.00um] (...)
    #
    # becomes:
    #
    # Nerve_ventral_root_T12_..._02
    #
    entity_names = names.str.split(
        r" \[",
        n=1,
        regex=True
    ).str[0]


    # ---------------------------------------------------------
    # Require the ventral-root discretized group
    # ---------------------------------------------------------
    #
    # This prevents duplicate trajectories from other groups,
    # e.g. femoral nerve discretizations.
    #
    ventral_root_group = names.str.contains(
        r"\("
        r"Discretized_Nerves/"
        r"Ventral_Root_Nerves/"
        r"Ventral_Root_Nerves_\(T7-L2\)_\d+um"
        r"\)\s*$",
        regex=True,
        na=False
    )


    # ---------------------------------------------------------
    # T7-T12
    #
    # Equivalent to:
    #
    # ('T7' ... 'T12' in i.Name)
    # and '_ventral_root_' in i.Name
    # ---------------------------------------------------------
    thoracic_mask = entity_names.str.contains(
        r"_ventral_root_T(?:7|8|9|10|11|12)_",
        regex=True,
        na=False
    )


    # ---------------------------------------------------------
    # L1-L2
    #
    # Equivalent to:
    #
    # 'Ventral_root_L1' or 'Ventral_root_L2'
    # 'cutaneous' not in i.Name
    # int(i.Name[-2:]) < 2
    # ---------------------------------------------------------
    lumbar_level_mask = entity_names.str.startswith(
        (
            "Ventral_root_L1",
            "Ventral_root_L2"
        )
    )


    noncutaneous_mask = ~entity_names.str.contains(
        "cutaneous",
        case=False,
        regex=False,
        na=False
    )


    # Extract final trajectory index:
    #
    # _01 -> 1
    # _02 -> 2
    #
    trajectory_number = pd.to_numeric(
        entity_names.str.extract(
            r"_(\d{2})$",
            expand=False
        ),
        errors="coerce"
    )


    lumbar_mask = (
        lumbar_level_mask
        & noncutaneous_mask
        & (trajectory_number < 2)
    )


    # ---------------------------------------------------------
    # Final mask
    # ---------------------------------------------------------
    ventral_root_mask = (
        ventral_root_group
        & (thoracic_mask | lumbar_mask)
    )


    return df.loc[ventral_root_mask].copy()

# Dorsal (Sensory) selection mask


def filter_dorsal_root_trajectory_rows(df, name_col="NeuronName"):
    """
    Select the same dorsal-root trajectories as the original Sim4Life
    entity-selection code.

    Selection:
        T7-T12:
            - name contains "_dorsal_root_T7_" ... "_dorsal_root_T12_"

        L1-L2:
            - name starts with "Dorsal_root_L1" or "Dorsal_root_L2"
            - exclude names containing "cutaneous"
            - trajectory index must be < 2 (i.e. _01 for current data)

        All rows must belong to:
            Discretized_Nerves/Dorsal_Root_Nerves/
            Dorsal_Root_Nerves_(T7-L2)_XXum

    Expected current result:
        138 rows total
        = 46 trajectories × 3 fibre diameters
    """

    if name_col not in df.columns:
        raise KeyError(
            f"Could not find column '{name_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    names = df[name_col].astype(str)

    # Extract the original Sim4Life entity/trajectory name.
    # Example:
    #
    # Nerve_dorsal_root_T12_nerve_twelfth_thoracic_right_02
    # [Sensory MRG Neuron 20.00um] (...)
    #
    # becomes:
    #
    # Nerve_dorsal_root_T12_nerve_twelfth_thoracic_right_02
    entity_names = names.str.split(r" \[", n=1, regex=True).str[0]


    # ---------------------------------------------------------
    # Must belong specifically to the dorsal-root discretized set
    # ---------------------------------------------------------
    dorsal_root_group = names.str.contains(
        r"\("
        r"Discretized_Nerves/"
        r"Dorsal_Root_Nerves/"
        r"Dorsal_Root_Nerves_\(T7-L2\)_\d+um"
        r"\)\s*$",
        regex=True,
        na=False
    )


    # ---------------------------------------------------------
    # T7-T12
    #
    # Equivalent to original:
    #   ('T7' ... 'T12' in i.Name)
    #   and '_dorsal_root_' in i.Name
    # ---------------------------------------------------------
    thoracic_mask = entity_names.str.contains(
        r"_dorsal_root_T(?:7|8|9|10|11|12)_",
        regex=True,
        na=False
    )


    # ---------------------------------------------------------
    # L1-L2
    #
    # Equivalent to original:
    #   'Dorsal_root_L1' or 'Dorsal_root_L2'
    #   'cutaneous' not in name
    #   int(i.Name[-2:]) < 2
    # ---------------------------------------------------------
    lumbar_level_mask = entity_names.str.startswith(
        ("Dorsal_root_L1", "Dorsal_root_L2")
    )

    noncutaneous_mask = ~entity_names.str.contains(
        "cutaneous",
        case=False,
        regex=False,
        na=False
    )

    # Extract final trajectory number, e.g.
    # "_01" -> 1
    trajectory_number = pd.to_numeric(
        entity_names.str.extract(r"_(\d{2})$", expand=False),
        errors="coerce"
    )

    lumbar_mask = (
        lumbar_level_mask
        & noncutaneous_mask
        & (trajectory_number < 2)
    )


    # ---------------------------------------------------------
    # Final selection
    # ---------------------------------------------------------
    dorsal_root_mask = (
        dorsal_root_group
        & (thoracic_mask | lumbar_mask)
    )

    return df.loc[dorsal_root_mask].copy()