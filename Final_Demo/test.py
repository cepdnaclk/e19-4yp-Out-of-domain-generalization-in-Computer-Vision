const promptPairs = [
            {
                negative: "No atypical cells infiltrating surrounding tissues",
                positive: "Atypical cells infiltrating surrounding tissues and disrupting normal structures",
                score: 90.13 // 0.9013 * 100 rounded
            },
            {
                negative: "No significant atypia in the surrounding lymphocytes", 
                positive: "Significant atypia observed in lymphocytes adjacent to tumor nests",
                score: 89.97 // 0.8997 * 100 rounded
            },
            {
                negative: "No evidence of fibrosis",
                positive: "Prominent stromal fibrosis surrounding tumor nests", 
                score: 89.94 // 0.8994 * 100 rounded
            },
            {
                negative: "Normal follicular architecture is preserved",
                positive: "Disrupted follicular architecture with loss of polarity",
                score: 89.40 // 0.894 * 100 rounded
            },
            {
                negative: "No prominent nucleoli are observed in lymphocytes",
                positive: "Cells exhibit large, prominent, and irregular nucleoli",
                score: 89.35 // 0.8935 * 100 rounded
            },
            {
                negative: "No giant cells or multinucleated cells are seen", 
                positive: "Presence of multinucleated giant cells, suggestive of specific tumor types",
                score: 88.84 // 0.8884 * 100 rounded
            },
            {
                negative: "No plasmacytoid differentiation is observed",
                positive: "Plasmacytoid differentiation is prominent within the tumor cells",
                score: 88.83 // 0.8883 * 100 rounded  
            },
            {
                negative: "No prominent nucleolus is seen",
                positive: "Large, prominent, and irregular nucleoli are present",
                score: 88.07 // 0.8807 * 100 rounded
            }