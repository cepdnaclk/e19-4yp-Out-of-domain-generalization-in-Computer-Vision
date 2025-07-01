INITIAL_CHATGPT_PROMPTS = [
    (
        "Uniform population of small lymphocytes with round nuclei and scant cytoplasm preserving nodal architecture",
        "Sheets of large pleomorphic cells with high nuclear-to-cytoplasmic ratio disrupting normal lymphoid architecture"
    ),
    (
        "Evenly spaced follicles with pale germinal centers and no evidence of cellular atypia",
        "Effacement of follicular architecture by monotonous population of atypical lymphoid cells with irregular nuclear contours"
    ),
    (
        "Fine, delicate trabecular stroma with clear sinusoids and minimal mitotic activity",
        "Dense, fibrotic stroma infiltrated by clusters of hyperchromatic tumor cells exhibiting frequent mitoses"
    ),
    (
        "Homogeneous distribution of small round lymphocytes with uniform chromatin and inconspicuous nucleoli",
        "Heterogeneous population of large cells with coarse chromatin, prominent nucleoli, and nuclear pleomorphism"
    ),
    (
        "Absence of necrosis and preservation of normal paracortical zones containing scattered immunoblasts",
        "Focal zones of coagulative necrosis surrounded by sheets of atypical lymphoid cells forming cohesive clusters"
    ),
    (
        "Delicate sinusoidal pattern with interspersed macrophages and no evidence of glandular structures",
        "Loss of sinusoidal pattern replaced by solid sheets of tumor cells forming abortive gland-like structures"
    ),
    (
        "Scant cytoplasm, small nuclei, and no prominent nucleoli in lymphoid cells",
        "Abundant eosinophilic cytoplasm, vesicular nuclei, and prominent nucleoli in malignant cells"
    ),
    (
        "Clear demarcation between cortex and medulla with well-formed germinal centers",
        "Blurring of cortical-medullary junction by infiltrative atypical cells permeating nodal tissue"
    ),
    (
        "Regularly spaced blood vessels with thin endothelial lining and no vascular proliferation",
        "Marked angiogenesis with proliferating endothelial cells and tumor cell nests encasing irregular vessels"
    ),
    (
        "Uniform staining pattern with light basophilic nuclei and no hyperchromasia",
        "Intensely hyperchromatic nuclei with coarse chromatin clumping and irregular chromatin distribution"
    ),
    (
        "Low nuclear-to-cytoplasmic ratio in lymphocytes with small, round, well-defined nuclei",
        "High nuclear-to-cytoplasmic ratio in tumor cells with large, irregular, and overlapping nuclei"
    ),
    (
        "Minimal variation in cell size, shape, and nuclear morphology across the field",
        "Marked anisocytosis and anisokaryosis with variation in size and shape of tumor cells and nuclei"
    ),
    (
        "Absence of multinucleated giant cells or Reed-Sternberg–like cells in the paracortex",
        "Presence of binucleated or multinucleated atypical cells with prominent nucleoli resembling Reed-Sternberg cells"
    ),
    (
        "Even distribution of lymphocytes without clusters of proliferating immunoblasts",
        "Confluent sheets of immunoblasts with open chromatin and prominent nucleoli forming clusters"
    ),
    (
        "Fine reticular network of stromal fibers with normal histiocyte distribution",
        "Dense collagenous stroma with frequent dust-like calcifications and infiltrating tumor cells"
    ),
    (
        "Sinusal pattern intact with histiocytes and macrophages prominently visible",
        "Sinusoidal spaces obliterated by solid sheets of malignant lymphoid cells"
    ),
    (
        "No perivascular cuffing or perisinusoidal infiltration by atypical cells",
        "Perivascular aggregates of large atypical cells forming cuff-like structures around vessels"
    ),
    (
        "Cytologically bland lymphocytes with smooth nuclear membranes and no nucleolar prominence",
        "Pleomorphic cells with irregular nuclear membranes, indentations, and conspicuous nucleoli"
    ),
    (
        "Regular arrangement of reticular fibers and no desmoplastic reaction",
        "Disorganized reticular framework replaced by dense desmoplastic reaction and tumor cell nests"
    ),
    (
        "Absence of mitotic figures or presence of only occasional normal mitoses",
        "Numerous atypical mitotic figures scattered throughout the field indicating high proliferative index"
    ),
    (
        "Uniform pale eosinophilic cytoplasm and small, round nuclei without vesicular changes",
        "Dense basophilic cytoplasm and vesicular nuclear changes with open chromatin in malignant cells"
    ),
    (
        "Normal lymph node parenchyma showing cortical lymphoid follicles and medullary cords",
        "Diffuse infiltration of medullary cords by large lymphoid cells with irregular chromatin distribution"
    ),
    (
        "Preserved capsule and subcapsular sinus without evidence of capsular invasion",
        "Capsular breach by aggregates of tumor cells extending beyond the node boundary"
    ),
    (
        "Scant interstitial edema and lack of inflammatory cell infiltrates beyond baseline",
        "Marked interstitial edema with mixed inflammatory infiltrate secondary to tumor-induced cytokine release"
    ),
    (
        "Smooth, well-defined nuclear borders and absence of nuclear grooves in lymphocytes",
        "Irregular nuclear borders with deep grooves and folds in malignant cells"
    ),
    (
        "No evidence of lymphoepithelial lesions or destruction of native tissue structures",
        "Lymphoepithelial lesions with invasion and destruction of normal nodal tissue by tumor cells"
    ),
    (
        "Normal sinus histiocytosis with scattered pale macrophages and no cellular atypia",
        "Proliferation of pale-staining histiocytes intermingled with atypical malignant cells"
    ),
    (
        "Evenly distributed chromatin with no coarse clumping or clearing",
        "Clumped or cleared chromatin with irregular distribution and chromatin margination in tumor nuclei"
    ),
    (
        "Absence of vascular proliferation and endothelial cell hyperplasia",
        "Marked endothelial hyperplasia with irregular, dilated vascular channels infiltrated by tumor cells"
    ),
    (
        "Regularly arranged reticulum cells with no cytoplasmic vacuolization",
        "Spindle-shaped tumor cells with vacuolated cytoplasm forming irregular fascicles"
    ),
    (
        "No presence of necrotic debris or karyorrhectic nuclear fragments",
        "Areas of karyorrhexis and necrotic debris amidst clusters of atypical cells"
    ),
    (
        "Lymphocytes exhibit uniform nuclear size and shape without membrane irregularities",
        "Tumor cells show nuclear membrane irregularities, indentations, and lobulations"
    ),
    (
        "Intact capsule with minimal subcapsular sinus expansion and absence of tumor nodules",
        "Distended subcapsular sinus filled with irregularly shaped tumor nodules protruding against the capsule"
    ),
    (
        "Well-defined germinal centers with tingible body macrophages and no atypical blast cells",
        "Germinal centers replaced by sheets of atypical blast-like cells lacking tingible body macrophages"
    ),
    (
        "No paracortical hyperplasia; paracortex composed of mature lymphocytes with few immunoblasts",
        "Paracortical expansion with clusters of immunoblasts and large atypical cells showing mitoses"
    ),
    (
        "Sinus histiocytes with pale cytoplasm interspersed among lymphocytes and no epithelial clusters",
        "Clusters of epithelial-like cells with eosinophilic cytoplasm forming alveolar patterns within the sinus"
    ),
    (
        "Regularly spaced nuclei with smooth chromatin and absent nucleolar prominence",
        "Irregular nuclear size with prominent nucleoli and clumped chromatin in tumor cells"
    ),
    (
        "No evidence of capsular or perinodal fat invasion by lymphoid cells",
        "Tumor cells breaching capsule invading perinodal adipose tissue forming irregular cell sheets"
    ),
    (
        "Absence of lymphovascular invasion; vessels clear of atypical cells",
        "Tumor emboli visible within lymphatic channels and small blood vessels"
    ),
    (
        "Uniform distribution of small, mature lymphocytes without variation in staining intensity",
        "Mixed staining intensity with hyperchromatic nuclei and pale cytoplasm in malignant cells"
    ),
    (
        "Preserved trabecular bone marrow interface without tumor cell infiltration",
        "Bone marrow interface disrupted by infiltration of atypical lymphoid cells"
    ),
    (
        "No fibrous bands traversing the node; delicate reticular meshwork present",
        "Thick fibrous bands splitting nodal parenchyma with tumor cells aligned along bands"
    ),
    (
        "Absence of Reed-Sternberg–like binucleated cells and lacunar histiocytes",
        "Presence of lacunar histiocytes and lacunar variant Reed-Sternberg cells amidst mixed infiltrate"
    ),
    (
        "Proliferation index (Ki-67) low with few cells in cell cycle",
        "Elevated proliferation index with numerous Ki-67–positive tumor nuclei indicating high mitotic activity"
    ),
    (
        "Regularly organized plasma cells in medullary cords with no plasmablastic features",
        "Sheets of large plasmablast-like cells with eccentric nuclei and prominent nucleoli replacing cords"
    ),
    (
        "Absence of necrotizing granulomas or histiocytic infiltration beyond normal range",
        "Necrotizing granulomas with central caseation and rim of atypical histiocytes and malignant lymphocytes"
    ),
    (
        "Uniform small germinal centers with polarized dark and light zones intact",
        "Disruption of germinal center polarization with monomorphic infiltrate and absent mantle zones"
    ),
    (
        "Normal follicular dendritic cell network present without regression or pale zones",
        "Disrupted follicular dendritic cell meshwork with regressive pale zones replaced by tumor cells"
    ),
    (
        "Intact capsular vessels with no perivascular tumor cell cuffing",
        "Perivascular cuffing of atypical cells around capsular and medullary vessels"
    ),
    (
        "Absence of sclerosis or broad fibrotic bands dividing nodal tissue",
        "Broad fibrotic bands creating nodular sclerosis pattern with tumor cells trapped in collagen"
    ),
    (
        "No evidence of epithelial infiltration; parenchyma devoid of cohesive epithelial clusters",
        "Cohesive clusters of epithelial-like tumor cells invading lymphoid parenchyma"
    ),
    (
        "Regular, small mitochondria-rich lymphocyte cytoplasm faintly basophilic",
        "Abundant pale, vacuolated cytoplasm in tumor cells with perinuclear clearing"
    ),
    (
        "Absence of apoptotic bodies and karyolysis among lymphoid cells",
        "Numerous apoptotic bodies and karyolytic nuclei indicating high turnover in malignant population"
    ),
    (
        "Fine granular chromatin with inconspicuous nucleoli across lymphocytes",
        "Coarse granular chromatin with conspicuous central nucleoli in neoplastic cells"
    )
]
